#!/usr/bin/env python3
"""Task 2 讨论点1：固定总 token 数，测 C16/32/64 的独立 MMA 求逆成本。

先读用途，再读代码：
1. 每个 CTA 只有一个 warp，在一个 kernel 内完成整个 Neumann 级数。
2. 直接使用 mma.sync.m16n8k16；四个 shared-memory 矩阵保存中间结果。
3. 这是统一实现策略下的实验，不是原 FlashKDA 寄存器 helper 的耗时。
4. 默认 T=8192、H=96，矩阵数=(T/C)*H；C 越大，矩阵数越少。
5. 先检查 GEMM 布局、求逆残差，再用 CUDA Graph/events 计时。

默认在已有 Slurm GPU 会话运行。--offline-check 只编译运行 CPU 分支，
核对 fragment 覆盖、参考与预算；它不模拟硬件 MMA 舍入，也不运行 GPU。

阅读顺序：lane 坐标 → gemm_shared → neumann_kernel → prepare → Graph 计时。
最后的 Python run 只负责编译、日志、SASS 导出和 SHA-256，不参与矩阵计算。
日志中 RAW_TIME 是独立采样，SUMMARY 是同一批次的中位数/范围。
摊销 ns/matrix 是吞吐归一化，不是一个矩阵的串行 latency。
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


# 源文件内用宏区分 CPU 与 CUDA，便于在无 nvcc 的环境核对公共逻辑。
CUDA_SOURCE = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef NEUMANN_MMA_OFFLINE_CHECK
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define HD __host__ __device__
#else
#define HD
#endif

using Matrix=std::vector<double>;
struct Coord { int row, col; };

// 【1. Warp fragment 坐标】按 PTX m16n8k16 的 FP16 布局显式装配寄存器。
// A 每 lane 8 个 half，B 每 lane 4 个 half，C/D 每 lane 4 个 half。
// a0/a1 打包为第一个32位寄存器，依次类推；行列均相对当前 MMA tile。
HD constexpr Coord a_coord(int lane,int i) {
  return {(lane/4)+((i/2)%2)*8,2*(lane%4)+(i%2)+(i/4)*8};
}
HD constexpr Coord b_coord(int lane,int i) {
  return {2*(lane%4)+(i%2)+(i/2)*8,lane/4};
}
HD constexpr Coord c_coord(int lane,int i) {
  return {lane/4+(i/2)*8,2*(lane%4)+(i%2)};
}

// 【2. 计时输入】每个矩阵有确定但不同的弱下三角元素，全部是精确二进制分数。
// |L_ij|<=1/(8N)，故 ||L||_inf<1/8，精确逆范数<=8/7。
// 此输入为隔离性能而构造，不代表真实 KDA 的 gate/keys 分布。
HD float input_value(int n,std::uint32_t matrix_id,int index) {
  int row=index/n,col=index%n;
  if(row<=col) return 0.0f;
  std::uint32_t x=(matrix_id+1u)*747796405u + std::uint32_t(index+1)*2891336453u;
  x=(x^(x>>16))*2246822519u; x^=x>>13;
  int signed_value=int((x>>16)%17u)-8;
  return float(signed_value)/(64.0f*n);
}

// 无 cuBLAS 的独立 FP64 前代参考，输入是 GPU 所用的同一份精确可表示 L。
Matrix reference(const Matrix& l,int n) {
  Matrix r(n*n,0.0);
  for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
    double sum=0;
    for(int k=0;k<i;++k) sum+=l[i*n+k]*r[k*n+j];
    r[i*n+j]=(i==j?1.0:0.0)-sum;
  }
  return r;
}
double inf_norm(const Matrix& a,int n) {
  double result=0;
  for(int i=0;i<n;++i) {
    double row=0; for(int j=0;j<n;++j) row+=std::abs(a[i*n+j]);
    result=std::max(result,row);
  }
  return result;
}
struct Error { int nonfinite=0,structure_bad=0; double residual=0,relative=0; };
Error errors(const Matrix& l,const Matrix& r,const Matrix& ref,int n) {
  Error e;
  for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
    if(!std::isfinite(r[i*n+j])) ++e.nonfinite;
    if((i==j && r[i*n+j]!=1.0) || (i<j && r[i*n+j]!=0.0)) ++e.structure_bad;
  }
  if(e.nonfinite) { e.residual=e.relative=INFINITY; return e; }
  Matrix residual(n*n),diff(n*n);
  for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
    double v=r[i*n+j]-(i==j?1.0:0.0);
    for(int k=0;k<i;++k) v+=l[i*n+k]*r[k*n+j];
    residual[i*n+j]=v; diff[i*n+j]=r[i*n+j]-ref[i*n+j];
  }
  e.residual=inf_norm(residual,n); e.relative=inf_norm(diff,n)/inf_norm(ref,n);
  return e;
}
int gemm_count(int n) { int stages=0; for(int p=2;p<n;p*=2) ++stages; return 2*stages; }
std::uint64_t matrix_flops(int n) { return std::uint64_t(gemm_count(n))*2*n*n*n; }

// GEMM 对照使用非对称、有正负的二进制分数；所有中间和均能被 FP16 精确表示。
// 它能检查 tile 的转置、寄存器排列、K 分块累加与输出坐标，不依赖容差掩盖错误。
double check_a(int i,int j) { return ((i*7+j*3+1)%5-2)/4.0; }
double check_b(int i,int j) { return ((i*5+j*11+3)%7-3)/4.0; }

#ifdef NEUMANN_MMA_OFFLINE_CHECK
// 【3A. 离线核对】覆盖检查只能发现漏项/重项；真实指令布局仍须 GPU GEMM 对照。
int main() {
  bool ok=true;
  std::puts("evidence=OFFLINE_LAYOUT_REFERENCE_BUDGET; GPU_executed=no; FP16_emulated=no");
  std::vector<int> a(256),b(128),c(128);
  for(int lane=0;lane<32;++lane) {
    for(int i=0;i<8;++i) { auto p=a_coord(lane,i); ++a[p.row*16+p.col]; }
    for(int i=0;i<4;++i) {
      auto p=b_coord(lane,i); ++b[p.row*8+p.col];
      auto q=c_coord(lane,i); ++c[q.row*8+q.col];
    }
  }
  for(auto* counts : {&a,&b,&c}) for(int count:*counts) ok=ok && count==1;
  std::printf("fragment_coverage_A256_B128_C128=%s\n",ok?"PASS":"FAIL");
  for(int n : {16,32,64}) {
    int batch=8192*96/n;
    Matrix l(n*n);
    for(int i=0;i<n*n;++i) l[i]=input_value(n,std::uint32_t(batch-1),i);
    Matrix ref=reference(l,n);
    Error e=errors(l,ref,ref,n);
    bool valid=inf_norm(l,n)<0.125 && e.residual<1e-12 && e.structure_bad==0;
    ok=ok && valid;
    // 验证 tiled 输出地址完整覆盖 N×N，并核对每个 K 切片累计的精确数学值。
    Matrix tiled(n*n,0.0),direct(n*n,0.0); std::vector<int> written(n*n,0);
    for(int mt=0;mt<n;mt+=16) for(int nt=0;nt<n;nt+=8)
      for(int lane=0;lane<32;++lane) for(int part=0;part<4;++part) {
        auto p=c_coord(lane,part); int row=mt+p.row,col=nt+p.col;
        ++written[row*n+col];
        for(int kt=0;kt<n;kt+=16) for(int k=0;k<16;++k)
          tiled[row*n+col]+=check_a(row,kt+k)*check_b(kt+k,col);
      }
    for(int i=0;i<n;++i) for(int j=0;j<n;++j)
      for(int k=0;k<n;++k) direct[i*n+j]+=check_a(i,k)*check_b(k,j);
    bool tiled_ok=tiled==direct;
    for(int count:written) tiled_ok=tiled_ok && count==1;
    ok=ok && tiled_ok;
    std::printf("CPU_CHECK,C=%d,L_inf=%.9g,ref_residual=%.9g,tile_scatter_math=%s\n",
                n,inf_norm(l,n),e.residual,tiled_ok?"PASS":"FAIL");
    std::printf("BUDGET,C=%d,matrices=%d,gemms_per_matrix=%d,warp_mma_per_matrix=%d,total_flop=%llu,io_MiB=%.0f,dynamic_smem_bytes=%d\n",
      n,batch,gemm_count(n),gemm_count(n)*(n/16)*(n/8)*(n/16),
      static_cast<unsigned long long>(matrix_flops(n)*batch),4.0*n*n*batch/(1<<20),8*n*n);
  }
  std::printf("offline_checks=%s; CUDA_compiled=no; CUDA_timed=no\n",ok?"PASS":"FAIL");
  return ok?0:3;
}
#else
#define CHECK(expr) do { auto e=(expr); if(e!=cudaSuccess) throw std::runtime_error( \
  std::string("CUDA line=")+std::to_string(__LINE__)+" "+cudaGetErrorString(e)); } while(0)

// 【3B. 实机计算】用16位原始位表示 shared memory 内的 half，打包时不做数值转换。
__device__ __forceinline__ std::uint32_t pack(std::uint16_t lo,std::uint16_t hi) {
  return std::uint32_t(lo)|(std::uint32_t(hi)<<16);
}

// 一个 warp 完成 N×N dense GEMM，A/B/C 都按行存储；C 不得覆盖 A 或 B。
// 所有32个 lane 必须一致参与；只有输出 tile 与 K 循环，没有按矩阵数值跳过工作。
template<int N>
__device__ __forceinline__ void gemm_shared(const std::uint16_t* a,const std::uint16_t* b,
                                           std::uint16_t* c,int lane) {
  // 保留外层循环以控制代码体积；C16/32/64 使用相同的装载/写回策略。
  #pragma unroll 1
  for(int mt=0;mt<N;mt+=16) {
    #pragma unroll 1
    for(int nt=0;nt<N;nt+=8) {
      std::uint32_t d0=0,d1=0;
      #pragma unroll
      for(int kt=0;kt<N;kt+=16) {
        std::uint32_t ar[4],br[2];
        #pragma unroll
        for(int reg=0;reg<4;++reg) {
          auto p=a_coord(lane,2*reg),q=a_coord(lane,2*reg+1);
          ar[reg]=pack(a[(mt+p.row)*N+kt+p.col],a[(mt+q.row)*N+kt+q.col]);
        }
        #pragma unroll
        for(int reg=0;reg<2;++reg) {
          auto p=b_coord(lane,2*reg),q=b_coord(lane,2*reg+1);
          br[reg]=pack(b[(kt+p.row)*N+nt+p.col],b[(kt+q.row)*N+nt+q.col]);
        }
        // A:4个寄存器；B:2个；C/D:2个。累加也为 FP16，沿 K 片顺序执行。
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
          : "+r"(d0),"+r"(d1)
          : "r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br[0]),"r"(br[1]));
      }
      #pragma unroll
      for(int part=0;part<4;++part) {
        auto p=c_coord(lane,part);
        std::uint32_t bits=part<2?d0:d1;
        c[(mt+p.row)*N+nt+p.col]=std::uint16_t(bits>>((part%2)*16));
      }
    }
  }
  // 后续阶段会由其他 lane 读取这些 shared-memory 写入，必须显式同步。
  __syncwarp();
}

template<int N>
__global__ void gemm_check_kernel(const std::uint16_t* ga,const std::uint16_t* gb,std::uint16_t* out) {
  extern __shared__ std::uint16_t s[];
  int lane=threadIdx.x; auto a=s,b=s+N*N,c=s+2*N*N;
  for(int i=lane;i<N*N;i+=32) { a[i]=ga[i]; b[i]=gb[i]; }
  __syncwarp(); gemm_shared<N>(a,b,c,lane);
  for(int i=lane;i<N*N;i+=32) out[i]=c[i];
}

template<int N>
__global__ void fill_input(std::uint16_t* input,std::size_t elements) {
  std::size_t at=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(at<elements) input[at]=__half_as_ushort(__float2half_rn(input_value(N,at/(N*N),at%(N*N))));
}

// 每个 CTA 独立求一个逆；共享内存为 power/next/R/temp 四矩阵，共8*N*N字节。
// 计时包含：读 L、构造 I-L、全部 GEMM/half add、同步、最终 BF16 写出。
// 它没有原 helper 的寄存器常驻和 ldmatrix/movmatrix 优化，不能当作原 kernel 延迟。
template<int N>
__global__ void neumann_kernel(const std::uint16_t* input,__nv_bfloat16* output) {
  extern __shared__ std::uint16_t s[];
  auto power=s,next=s+N*N,r=s+2*N*N,temp=s+3*N*N;
  int lane=threadIdx.x;
  std::size_t base=std::size_t(blockIdx.x)*N*N;
  for(int i=lane;i<N*N;i+=32) {
    power[i]=input[base+i];
    r[i]=__half_as_ushort(__float2half_rn((i/N==i%N?1.0f:0.0f)-__half2float(__ushort_as_half(power[i]))));
  }
  __syncwarp();
  #pragma unroll 1
  for(int p=2;p<N;p*=2) {
    gemm_shared<N>(power,power,next,lane);
    auto swap=power; power=next; next=swap;
    gemm_shared<N>(r,power,temp,lane);
    for(int i=lane;i<N*N;i+=32)
      r[i]=__half_as_ushort(__hadd(__ushort_as_half(r[i]),__ushort_as_half(temp[i])));
    __syncwarp();
  }
  for(int i=lane;i<N*N;i+=32)
    output[base+i]=__float2bfloat16_rn(__half2float(__ushort_as_half(r[i])));
}

// GPU 布局对照必须是精确匹配，失败时不继续把错误实现拿去计时。
template<int N> void validate_gemm(cudaStream_t stream) {
  std::vector<std::uint16_t> a(N*N),b(N*N),out(N*N);
  for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
    a[i*N+j]=__half_as_ushort(__float2half_rn(float(check_a(i,j))));
    b[i*N+j]=__half_as_ushort(__float2half_rn(float(check_b(i,j))));
  }
  std::uint16_t *da=nullptr,*db=nullptr,*dc=nullptr;
  CHECK(cudaMalloc(reinterpret_cast<void**>(&da),2*N*N));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&db),2*N*N));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&dc),2*N*N));
  CHECK(cudaMemcpyAsync(da,a.data(),2*N*N,cudaMemcpyHostToDevice,stream));
  CHECK(cudaMemcpyAsync(db,b.data(),2*N*N,cudaMemcpyHostToDevice,stream));
  gemm_check_kernel<N><<<1,32,6*N*N,stream>>>(da,db,dc); CHECK(cudaGetLastError());
  CHECK(cudaMemcpyAsync(out.data(),dc,2*N*N,cudaMemcpyDeviceToHost,stream));
  CHECK(cudaStreamSynchronize(stream));
  int wrong=0; double max_error=0;
  for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
    double expected=0; for(int k=0;k<N;++k) expected+=check_a(i,k)*check_b(k,j);
    double observed=__half2float(__ushort_as_half(out[i*N+j]));
    if(observed!=expected) ++wrong;
    max_error=std::max(max_error,std::abs(observed-expected));
  }
  std::printf("GEMM_CHECK,C=%d,wrong_elements=%d,max_abs_error=%.9g\n",N,wrong,max_error);
  CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
  if(wrong) throw std::runtime_error("MMA layout/GEMM check failed; timing skipped");
}

struct Case {
  int n=0,batch=0;
  std::uint16_t* input=nullptr;
  __nv_bfloat16* output=nullptr;
  cudaGraph_t graph=nullptr;
  cudaGraphExec_t executable=nullptr;
  std::vector<double> samples;
};

// 只检查首/中/末三个实际批次位置，明确为抽样，不冒称全部矩阵或KDA正确性。
void validate_inverse(const Case& c,cudaStream_t stream,const char* phase) {
  for(int id : {0,c.batch/2,c.batch-1}) {
    Matrix l(c.n*c.n),r(c.n*c.n);
    std::vector<__nv_bfloat16> output(c.n*c.n);
    CHECK(cudaMemcpyAsync(output.data(),c.output+std::size_t(id)*c.n*c.n,
                          2*c.n*c.n,cudaMemcpyDeviceToHost,stream));
    CHECK(cudaStreamSynchronize(stream));
    for(int i=0;i<c.n*c.n;++i) { l[i]=input_value(c.n,id,i); r[i]=__bfloat162float(output[i]); }
    Matrix ref=reference(l,c.n); Error e=errors(l,r,ref,c.n),er=errors(l,ref,ref,c.n);
    // 0.002是本微基准的实现筛查阈值；不是模型精度或压力输入的统一验收标准。
    bool ok=e.nonfinite==0 && e.structure_bad==0 && e.residual<=0.002 && e.relative<=0.002 && er.residual<1e-12;
    std::printf("INVERSE_CHECK,C=%d,matrix_id=%d,phase=%s,nonfinite=%d,structure_bad=%d,residual_inf=%.9g,relative_inf=%.9g,ref_residual=%.9g,status=%s\n",
                c.n,id,phase,e.nonfinite,e.structure_bad,e.residual,e.relative,er.residual,ok?"PASS":"FAIL");
    if(!ok) throw std::runtime_error("inverse check failed; timing skipped");
  }
}

// 准备所有分配、输入、正确性对照和 Graph。计时不包含这些操作。
template<int N> Case prepare(int total,int graph_iterations,cudaStream_t stream) {
  validate_gemm<N>(stream);
  Case c; c.n=N; c.batch=total/N;
  std::size_t elements=std::size_t(c.batch)*N*N,bytes=elements*2;
  CHECK(cudaMalloc(reinterpret_cast<void**>(&c.input),bytes));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&c.output),bytes));
  fill_input<N><<<(elements+255)/256,256,0,stream>>>(c.input,elements); CHECK(cudaGetLastError());
  // 统一请求最大 shared-memory carveout；这是 hint，实际配置仍由运行时决定。
  CHECK(cudaFuncSetAttribute(neumann_kernel<N>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
  cudaFuncAttributes attr{};
  CHECK(cudaFuncGetAttributes(&attr,neumann_kernel<N>));
  int active=0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,neumann_kernel<N>,32,8*N*N));
  std::printf("RESOURCE,C=%d,threads=32,registers_per_thread=%d,static_smem=%zu,dynamic_smem=%d,local_bytes_per_thread=%zu,occupancy_api_blocks_per_SM=%d,carveout_hint=100\n",
              N,attr.numRegs,attr.sharedSizeBytes,8*N*N,attr.localSizeBytes,active);
  std::printf("BUDGET,C=%d,matrices=%d,gemms_per_matrix=%d,warp_mma_per_matrix=%d,total_flop=%llu,io_MiB=%.0f\n",
              N,c.batch,gemm_count(N),gemm_count(N)*(N/16)*(N/8)*(N/16),
              static_cast<unsigned long long>(matrix_flops(N)*c.batch),2.0*bytes/(1<<20));
  neumann_kernel<N><<<c.batch,32,8*N*N,stream>>>(c.input,c.output); CHECK(cudaGetLastError());
  validate_inverse(c,stream,"direct_launch");
  CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  for(int i=0;i<graph_iterations;++i)
    neumann_kernel<N><<<c.batch,32,8*N*N,stream>>>(c.input,c.output);
  CHECK(cudaStreamEndCapture(stream,&c.graph));
  CHECK(cudaGetLastError());
  std::size_t node_count=0;
  CHECK(cudaGraphGetNodes(c.graph,nullptr,&node_count));
  if(node_count!=std::size_t(graph_iterations)) throw std::runtime_error("unexpected Graph node count");
  CHECK(cudaGraphInstantiateWithFlags(&c.executable,c.graph,0));
  std::printf("GRAPH,C=%d,kernel_nodes=%zu\n",N,node_count);
  return c;
}

void cleanup(Case& c) {
  CHECK(cudaGraphExecDestroy(c.executable)); CHECK(cudaGraphDestroy(c.graph));
  CHECK(cudaFree(c.input)); CHECK(cudaFree(c.output));
}

// 【4. 计时入口】先完成所有尺寸的检查和预热，再轮换尺寸顺序采样，降低顺序偏差。
int main(int argc,char** argv) {
  // stdout 经 Python 管道接收；逐行刷新，若运行失败仍能留下已完成阶段的证据。
  std::setvbuf(stdout,nullptr,_IOLBF,0);
  try {
    if(argc!=6) throw std::runtime_error("expected: tokens heads graph_iterations rounds warmup_graphs");
    int t=std::stoi(argv[1]),h=std::stoi(argv[2]),iters=std::stoi(argv[3]),rounds=std::stoi(argv[4]),warmup=std::stoi(argv[5]);
    if(t<=0 || t%64 || h<=0 || std::int64_t(t)*h>16777216 || iters<1 || iters>100 || rounds<3 || rounds>101 || warmup<1 || warmup>100)
      throw std::runtime_error("parameters outside supported bounds");
    int device=0,driver=0,runtime=0; CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop,device));
    CHECK(cudaDriverGetVersion(&driver)); CHECK(cudaRuntimeGetVersion(&runtime));
    std::puts("evidence=REAL_MMA_NEUMANN_BATCH_BENCHMARK");
    std::printf("device=%s capability=%d.%d SMs=%d driver=%d runtime=%d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount,driver,runtime);
    std::puts("scope=standalone_one_warp_four_smem_matrices; direct_FP16_mma_sync; BF16_output; not_original_FlashKDA; not_instruction_latency");
    std::puts("timed=Graph_kernel_batches_with_global_IO_smem_sync_and_conversion; no_cuBLAS; reused_buffers; no_cache_flush; clocks_unmodified");
    std::printf("T=%d H=%d graph_iterations=%d rounds=%d warmup_graphs=%d input_L_inf_bound=0.125\n",t,h,iters,rounds,warmup);
    cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::vector<Case> cases;
    cases.push_back(prepare<16>(t*h,iters,stream));
    cases.push_back(prepare<32>(t*h,iters,stream));
    cases.push_back(prepare<64>(t*h,iters,stream));
    for(auto& c:cases) {
      // 擦掉预检输出，确认后续 Graph 真正重写结果；操作在计时之外。
      CHECK(cudaMemsetAsync(c.output,0xff,std::size_t(c.batch)*c.n*c.n*2,stream));
      for(int i=0;i<warmup;++i) CHECK(cudaGraphLaunch(c.executable,stream));
      validate_inverse(c,stream,"graph_warmup");
    }
    std::puts("all_prechecks=PASS; timing_starts_after_this_line");
    cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    for(int round=0;round<rounds;++round) for(int slot=0;slot<3;++slot) {
      auto& c=cases[(round+slot)%3];
      CHECK(cudaEventRecord(start,stream));
      CHECK(cudaGraphLaunch(c.executable,stream));
      CHECK(cudaEventRecord(stop,stream)); CHECK(cudaEventSynchronize(stop));
      float graph_ms=0; CHECK(cudaEventElapsedTime(&graph_ms,start,stop));
      double batch_ms=double(graph_ms)/iters;
      if(!std::isfinite(batch_ms) || batch_ms<=0) throw std::runtime_error("invalid timing");
      c.samples.push_back(batch_ms);
      std::printf("RAW_TIME,C=%d,round=%d,order_slot=%d,graph_ms=%.9g,batch_ms=%.9g\n",c.n,round,slot,double(graph_ms),batch_ms);
    }
    double base=0;
    for(auto& c:cases) {
      auto values=c.samples; std::sort(values.begin(),values.end());
      double median=values.size()%2?values[values.size()/2]:0.5*(values[values.size()/2-1]+values[values.size()/2]);
      if(c.n==16) base=median;
      std::printf("SUMMARY,C=%d,matrices=%d,median_batch_ms=%.9g,min_batch_ms=%.9g,max_batch_ms=%.9g,ratio_to_C16=%.9g,amortized_ns_per_matrix=%.9g\n",
                  c.n,c.batch,median,values.front(),values.back(),median/base,median*1e6/c.batch);
    }
    std::puts("timing_completed=yes; amortized_ns_is_throughput_normalization_not_serial_latency");
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    for(auto& c:cases) cleanup(c);
    CHECK(cudaStreamDestroy(stream));
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"ERROR: %s\n",e.what()); return 2;
  }
}
#endif
'''


# 【5. Python 运行与证据】旧探针和旧日志不变，本实验始终创建新的 UTC 路径。
def run(args):
    work = args.work_dir.resolve()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    suffix = 'offline_task2_neumann_mma' if args.offline_check else 'task2_neumann_mma'
    raw = work / 'logs/raw'
    build = work / 'build' / f'{run_id}_{suffix}'
    raw.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=False)
    log_path = raw / f'{run_id}_{suffix}.log'
    before = raw / f'{run_id}_{suffix}_gpu_before.csv'
    after = raw / f'{run_id}_{suffix}_gpu_after.csv'
    source, binary = build / 'neumann_mma.cu', build / 'neumann_mma'
    sass = raw / f'{run_id}_{suffix}_sass.txt'
    artifacts = [Path(__file__).resolve(), source, binary, sass, before, after, log_path]
    status, attempted_gpu = 1, False
    with log_path.open('x') as log:
        def emit(value):
            print(value, flush=True)
            log.write(value + '\n')
            log.flush()

        # 不使用 shell 拼接执行命令，完整保留 stdout/stderr 与真实退出码。
        def command(argv, timeout, output_file=None):
            emit('command=' + shlex.join([str(x) for x in argv]))
            p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=timeout, check=False)
            if output_file is None:
                if p.stdout:
                    emit(p.stdout.rstrip('\n'))
            else:
                output_file.write_text(p.stdout)
                emit(f'command_output_file={output_file}; lines={len(p.stdout.splitlines())}')
            emit(f'command_exit={p.returncode}')
            if p.returncode:
                raise RuntimeError(f'command failed with exit {p.returncode}')
            return p.stdout

        def snapshot(path):
            p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                '--format=csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=20, check=False)
            path.write_text(p.stdout)
            emit(f'gpu_snapshot={path}')
            emit(p.stdout.rstrip('\n'))
            if p.returncode:
                raise RuntimeError('nvidia-smi failed')

        emit('purpose=Task2_discussion1_MMA_Neumann_batch_cost; not_FlashKDA_benchmark')
        emit(f'mode={"OFFLINE_CPU_ONLY" if args.offline_check else "REAL_CUDA_PENDING"}')
        emit(f'utc_run_id={run_id}')
        emit(f'python={sys.executable}')
        emit(f'script={Path(__file__).resolve()}')
        emit(f'slurm_job_id={os.environ.get("SLURM_JOB_ID", "missing")}')
        emit(f'raw_log={log_path}')
        try:
            source.write_text(CUDA_SOURCE)
            if args.offline_check:
                cxx = shutil.which('g++')
                if cxx is None:
                    raise RuntimeError('g++ unavailable')
                command([cxx, '--version'], 20)
                command([cxx, '-x', 'c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-pedantic',
                         '-ffp-contract=off', '-DNEUMANN_MMA_OFFLINE_CHECK', str(source), '-o', str(binary)], 60)
                command([str(binary)], 20)
            else:
                if not os.environ.get('SLURM_JOB_ID'):
                    raise RuntimeError('Run inside the existing Slurm GPU allocation.')
                if not args.nvcc.is_file() or not os.access(args.nvcc, os.X_OK):
                    raise RuntimeError(f'nvcc not executable: {args.nvcc}')
                if shutil.which('nvidia-smi') is None:
                    raise RuntimeError('nvidia-smi unavailable')
                command([str(args.nvcc), '--version'], 20)
                command([str(args.nvcc), '-std=c++17', '-O3', '-arch=sm_103a', '--fmad=false',
                         '-Xcompiler=-ffp-contract=off', '-Xptxas=-v', str(source), '-o', str(binary)], 240)
                # SASS 来自本次编译的文件，导出不运行 GPU；只打印统计，全文另存。
                cuobjdump = args.nvcc.parent / 'cuobjdump'
                if cuobjdump.is_file() and os.access(cuobjdump, os.X_OK):
                    listing = command([str(cuobjdump), '--dump-sass', str(binary)], 60, sass)
                    opcodes = re.findall(r'/\*[0-9a-fA-F]+\*/\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)', listing)
                    variants = Counter(op for op in opcodes if op.startswith('HMMA.'))
                    emit('HMMA_static_variants=' + repr(dict(sorted(variants.items()))))
                    emit('SASS_counts_are_static_lines_not_dynamic_MMA_counts')
                    if not variants:
                        raise RuntimeError('no HMMA opcode parsed from this binary; review SASS before timing')
                else:
                    emit('sass_export=unavailable; cuobjdump_not_found; PTX_source_scope_only')
                snapshot(before)
                attempted_gpu = True
                command([str(binary), str(args.tokens), str(args.heads), str(args.graph_iterations),
                         str(args.rounds), str(args.warmup_graphs)], 120)
            status = 0
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            emit(f'ERROR: {exc}')
            if isinstance(exc, subprocess.TimeoutExpired) and exc.stdout:
                emit(exc.stdout.decode(errors='replace') if isinstance(exc.stdout, bytes) else exc.stdout)
        finally:
            if attempted_gpu:
                try:
                    snapshot(after)
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    emit(f'ERROR: after snapshot: {exc}')
                    status = 1
            emit(f'benchmark_exit={status}')
            emit('scope=standalone_shared_memory_MMA_inverse; original_FlashKDA_C32_C64_not_implemented')
    manifest = log_path.with_name(log_path.name + '.sha256')
    with manifest.open('x') as hashes:
        for path in artifacts:
            if path.is_file():
                row = f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}'
                print(row)
                hashes.write(row + '\n')
    print(f'raw_log={log_path}')
    print(f'hash_manifest={manifest}')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline-check', action='store_true', help='只编译运行 CPU 布局/参考/预算核对')
    parser.add_argument('--work-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'runs')
    parser.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    parser.add_argument('--tokens', type=int, default=8192)
    parser.add_argument('--heads', type=int, default=96)
    parser.add_argument('--graph-iterations', type=int, default=5, help='一个 Graph 内串行执行的完整 grid 数')
    parser.add_argument('--rounds', type=int, default=9, help='各 C 的独立采样数；9轮使三种顺序各出现3次')
    parser.add_argument('--warmup-graphs', type=int, default=3, help='计时前，每种 C 的 Graph 预热次数')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
