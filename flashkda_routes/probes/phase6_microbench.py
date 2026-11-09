#!/usr/bin/env python3
"""Task 2 讨论点2：SM80 与 tcgen05 的 CHUNK=16 / Phase 6 形状对比。

先读实验范围：
1. 数学工作为 A[128,16] @ B[16,128]，再做 S=BF16(FMA(S,g,A@B))。
2. 对比一次更新和连续32次更新；每CTA一次TMEM分配/释放计入总时间。
3. SM80用ldmatrix+mma.sync，tcgen05用descriptor+TMEM+commit/wait/load。
4. 两边共享同一packed输入、列主序state和BF16舍入规则；先完整对拍再计时。
5. 固定A/B在32步中复用，未包含原K2前序阶段产生U的过程，不是完整FlashKDA。

阅读顺序：packed/descriptor → fixture/reference → phase6_sm80 → phase6_tcgen
→ validate_product/check_state → CUDA Graph计时 → Python日志与哈希。

默认在已有Slurm GPU会话运行；--offline-check只用CPU核对，不启动GPU。
输出：本轮独立UTC构建目录、完整日志、SASS、前后GPU快照及SHA256。
RAW_TIME是每轮原始时间；COMPARISON只比较同一blocks/steps。
normalized_update_batch_us是批次时间除以步数，不是单条指令或单CTA延迟。
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

CUDA_SOURCE = r'''
// Task 2 讨论点2：CHUNK=16，独立 Phase 6 形状的完整状态更新。
// 本文件嵌入 Python 交付脚本；CPU 分支只能核对布局/参考，不能证明 CUDA 通过。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef PHASE6_OFFLINE
#include <cuda_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif

constexpr int M=128, N=128, K=16, Threads=128, Elements=M*N;
// A 与转置存放的 B 使用同一 K-major 无 swizzle 的 8×8 core 布局。
// 以 BF16 元素计：一组8行有128元素；K的后8项距前8项64元素。
// descriptor 因此使用 LBO=128字节、SBO=256字节，不能当普通行主序。
HD constexpr int packed(int row,int k) {
  return (row/8)*128+(k/8)*64+(row%8)*8+k%8;
}
// state 用列主序。两边相同，方便 TMEM 的连续行读出；不是原完整 K2 布局。
HD constexpr int state_at(int row,int col) { return col*M+row; }
HD constexpr std::uint64_t descriptor(std::uint32_t address) {
  return ((address&0x3ffffu)>>4) | (std::uint64_t(128/16)<<16)
       | (std::uint64_t(256/16)<<32) | (std::uint64_t(1)<<46);
}
// dtype=FP32；atype=btype=BF16；不转置K-major描述符；M/16=8，N/8=16。
constexpr std::uint32_t IDesc=(1u<<4)|(1u<<7)|(1u<<10)|((N/8u)<<17)|((M/16u)<<24);

// 【输入与 CPU 参考】固定可重现的二进制分数，避免用宽松误差容限掩盖布局错误。
// A/B绝对值<=1/4，16项乘积和绝对值<=1，FP32可精确计算这些测试点积。
HD std::uint32_t mix(std::uint32_t x) {
  x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
HD float fixture(int kind,int id,int row,int col) {
  std::uint32_t x=mix((std::uint32_t(id)+1u)*0x9e3779b9u
      +std::uint32_t(row)*131u+std::uint32_t(col)*8191u+std::uint32_t(kind)*104729u);
  return float(int(x%17u)-8)/(kind==2?16.0f:32.0f);
}
HD float gate(int id,int row) { return float(4+((id+row)%5))/8.0f; }

float from_bf16(std::uint16_t x) {
  std::uint32_t bits=std::uint32_t(x)<<16; float f; std::memcpy(&f,&bits,4); return f;
}
std::uint16_t to_bf16(float f) {
  // 当前参考仅处理有限值：round-to-nearest，tie-to-even。
  std::uint32_t bits; std::memcpy(&bits,&f,4);
  return std::uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
float product_reference(int id,int row,int col) {
  double sum=0;
  for(int k=0;k<K;++k) sum+=double(fixture(0,id,row,k))*fixture(1,id,col,k);
  return float(sum);
}
std::uint16_t update_reference(int id,int row,int col,int steps,float product) {
  auto state=to_bf16(fixture(2,id,row,col));
  for(int step=0;step<steps;++step)
    state=to_bf16(std::fma(from_bf16(state),gate(id,row),product));
  return state;
}

// 共40KiB矩阵+512B gate+16B同步元数据，再按128字节对齐。
// 为可比性，两实现使用同一动态 shared-memory 大小，包括 SM80 未用的元数据。
struct alignas(128) Shared {
  std::uint16_t a[M*K],bt[N*K],state[Elements];
  float g[M];
  alignas(8) std::uint64_t barrier;
  std::uint32_t tmem,unused;
};
static_assert(sizeof(Shared)==41600,"unexpected shared storage layout");

#ifdef PHASE6_OFFLINE
int main() {
  bool ok=true;
  std::puts("evidence=OFFLINE_LAYOUT_AND_REFERENCE; CUDA_compiled=no; GPU_executed=no");
  std::vector<int> seen(M*K),output_sm80(Elements),output_tmem(Elements);
  for(int r=0;r<M;++r) for(int k=0;k<K;++k) {
    ++seen[packed(r,k)];
    // 按 descriptor 的独立字节公式核对所有 core 元素的实际位置。
    int address=(r/8)*256+(k/8)*128+(r%8)*16+(k%8)*2;
    ok=ok && address==2*packed(r,k);
  }
  for(int count:seen) ok=ok && count==1;
  // HMMA每warp负责32列；每个m16n8 accumulator按4个FP32寄存器散布。
  for(int w=0;w<4;++w) for(int mt=0;mt<M;mt+=16) for(int nt=0;nt<32;nt+=8)
    for(int lane=0;lane<32;++lane) for(int part=0;part<4;++part) {
      int row=mt+lane/4+(part/2)*8,col=w*32+nt+(lane%4)*2+part%2;
      ++output_sm80[state_at(row,col)];
    }
  for(int w=0;w<4;++w) for(int lane=0;lane<32;++lane)
    for(int col=0;col<N;++col) ++output_tmem[state_at(w*32+lane,col)];
  for(int i=0;i<Elements;++i) ok=ok && output_sm80[i]==1 && output_tmem[i]==1;
  // ldmatrix.x4/x2 的每组8线程提供一个8x8子矩阵的行地址；这里只核对地址。
  for(int mt=0;mt<M;mt+=16) for(int lane=0;lane<32;++lane) {
    int row=mt+lane%8+((lane/8)%2)*8,k=(lane/16)*8;
    ok=ok && (2*packed(row,k))%16==0;
  }
  ok=ok && ((IDesc>>24)&31)==8 && ((IDesc>>17)&63)==16;
  ok=ok && ((descriptor(0)>>46)&7)==1 && ((descriptor(0)>>16)&0x3fff)==8;
  std::printf("layout_coverage_and_descriptor=%s; hardware_mapping_still_needs_GPU_check\n",ok?"PASS":"FAIL");
  // 两个半途舍入点分别应向偶数末位舍入，检查手写BF16参考的关键分支。
  ok=ok && to_bf16(1.00390625f)==0x3f80 && to_bf16(1.01171875f)==0x3f82;
  for(int id : {0,48,95}) for(int steps : {1,32}) {
    double checksum=0,max_abs=0;
    for(int row=0;row<M;++row) for(int col=0;col<N;++col) {
      float product=product_reference(id,row,col);
      float state=from_bf16(update_reference(id,row,col,steps,product));
      ok=ok && std::isfinite(state) && std::abs(product)<=1;
      checksum+=state; max_abs=std::max(max_abs,std::abs(double(state)));
    }
    std::printf("CPU_REFERENCE,id=%d,steps=%d,checksum=%.12g,max_abs=%.9g\n",id,steps,checksum,max_abs);
  }
  std::printf("BUDGET,M=%d,N=%d,K=%d,threads=%d,shared_bytes=%zu,tmem_columns=128,mma_flop_per_update=524288,epilogue_flop_per_update=32768,idesc=0x%08x\n",M,N,K,Threads,sizeof(Shared),IDesc);
  std::printf("offline_check=%s; CUDA_timing=not_performed\n",ok?"PASS":"FAIL");
  return ok?0:3;
}
#else
#define CUDA_OK(call) do { auto e=(call); if(e!=cudaSuccess) throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e)); } while(0)
__device__ float bf(std::uint16_t x) { return __uint_as_float(std::uint32_t(x)<<16); }
__device__ std::uint16_t rn_bf(float x) {
  std::uint16_t out; asm("cvt.rn.bf16.f32 %0, %1;":"=h"(out):"f"(x)); return out;
}
__device__ std::uint32_t smem_address(const void* p) { return std::uint32_t(__cvta_generic_to_shared(p)); }

struct Inputs {
  const std::uint16_t *a,*bt,*s;
  const float *g;
};
template<bool Update> __device__ void stage_inputs(Shared& s,Inputs in) {
  int id=blockIdx.x,tid=threadIdx.x;
  for(int i=tid;i<M*K;i+=Threads) {
    s.a[i]=in.a[id*M*K+i]; s.bt[i]=in.bt[id*N*K+i];
  }
  if constexpr(Update) {
    for(int i=tid;i<Elements;i+=Threads) s.state[i]=in.s[id*Elements+i];
    s.g[tid]=in.g[id*M+tid];
  }
}

// 【SM80】A用ldmatrix.x4，B^T用ldmatrix.x2；每warp独占32列。
// B fragment 在连续更新中常驻寄存器。A每轮按m16行块读取并复用于四个n8块。
__device__ void load_a(const std::uint16_t* a,int mt,int lane,std::uint32_t (&r)[4]) {
  int row=mt+lane%8+((lane/8)%2)*8,k=(lane/16)*8;
  auto addr=smem_address(a+packed(row,k));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(addr):"memory");
}
__device__ void load_b(const std::uint16_t* b,int nt,int lane,std::uint32_t (&r)[2]) {
  int row=nt+lane%8,k=((lane/8)%2)*8;
  auto addr=smem_address(b+packed(row,k));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
    :"=r"(r[0]),"=r"(r[1]):"r"(addr):"memory");
}
template<bool Update> __global__ void phase6_sm80(Inputs in,std::uint16_t* out,float* product,int steps) {
  extern __shared__ __align__(128) unsigned char storage[];
  Shared& s=*reinterpret_cast<Shared*>(storage);
  int tid=threadIdx.x,lane=tid%32,warp=tid/32,id=blockIdx.x;
  stage_inputs<Update>(s,in); __syncthreads();
  std::uint32_t breg[4][2];
  #pragma unroll
  for(int nb=0;nb<4;++nb) load_b(s.bt,warp*32+nb*8,lane,breg[nb]);
  #pragma unroll 1
  for(int step=0;step<steps;++step) {
    #pragma unroll 1
    for(int mt=0;mt<M;mt+=16) {
      std::uint32_t a[4]; load_a(s.a,mt,lane,a);
      #pragma unroll
      for(int nb=0;nb<4;++nb) {
        float d[4]={0,0,0,0};
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
          :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(breg[nb][0]),"r"(breg[nb][1]));
        #pragma unroll
        for(int p=0;p<4;++p) {
          int row=mt+lane/4+(p/2)*8,col=warp*32+nb*8+(lane%4)*2+p%2;
          int at=state_at(row,col);
          if constexpr(Update) s.state[at]=rn_bf(fmaf(bf(s.state[at]),s.g[row],d[p]));
          else product[id*Elements+at]=d[p];
        }
      }
    }
    __syncthreads();
  }
  if constexpr(Update)
    for(int i=tid;i<Elements;i+=Threads) out[id*Elements+i]=s.state[i];
}

// 【tcgen05】分配/释放由warp0集体执行，MMA/commit仅由thread0发起。
// mbarrier负责MMA完成；tcgen05.ld后wait::ld；CTA同步前后补tcgen05 fence。
__device__ void wait_mma(std::uint32_t barrier,int phase) {
  for(int tries=0;;++tries) {
    std::uint32_t done;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; "
                 "selp.u32 %0,1,0,p; }":"=r"(done):"r"(barrier),"r"(phase):"memory");
    if(done) break;
    // 实现错误时终止本CUDA上下文，避免无限等待；失败不进入后续计时。
    if(tries==1000000) asm volatile("trap;");
  }
}
__device__ void load_tmem16(std::uint32_t address,float (&out)[16]) {
  // b32寄存器先接收原始位，再零成本解释为FP32，避免PTX操作数类型歧义。
  std::uint32_t r[16];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
     "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
    :"r"(address):"memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
  #pragma unroll
  for(int i=0;i<16;++i) out[i]=__uint_as_float(r[i]);
}
template<bool Update> __global__ void phase6_tcgen(Inputs in,std::uint16_t* out,float* product,int steps) {
  extern __shared__ __align__(128) unsigned char storage[];
  Shared& s=*reinterpret_cast<Shared*>(storage);
  int tid=threadIdx.x,warp=tid/32,id=blockIdx.x;
  auto bar=smem_address(&s.barrier);
  if(warp==0) {
    auto address=smem_address(&s.tmem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"::"r"(address):"memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;":::"memory");
  }
  if(tid==0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(bar):"memory");
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  stage_inputs<Update>(s,in);
  // generic stores必须先对异步MMA可见；所有写入线程都执行proxy fence。
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  __syncthreads();
  auto tmem=s.tmem;
  auto ad=descriptor(smem_address(s.a)),bd=descriptor(smem_address(s.bt));
  #pragma unroll 1
  for(int step=0;step<steps;++step) {
    asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
    if(tid==0) {
      // p=false：每轮只计算A*B，不能把上一轮TMEM product累加进来。
      asm volatile("{ .reg .pred p; setp.ne.u32 p,0,0; "
        "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,{%4,%4,%4,%4},p; }"
        ::"r"(tmem),"l"(ad),"l"(bd),"r"(IDesc),"r"(0):"memory");
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(bar):"memory");
    }
    wait_mma(bar,step&1);
    asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
    // 每warp读其32条TMEM lane；taddr在warp内必须一致，不加lane_id。
    #pragma unroll 1
    for(int col=0;col<N;col+=16) {
      float values[16];
      load_tmem16(tmem+(std::uint32_t(warp*32)<<16)+col,values);
      #pragma unroll
      for(int j=0;j<16;++j) {
        int at=state_at(tid,col+j);
        if constexpr(Update) s.state[at]=rn_bf(fmaf(bf(s.state[at]),s.g[tid],values[j]));
        else product[id*Elements+at]=values[j];
      }
    }
    // 下一轮thread0写TMEM前，必须保证其他warp的读已经完成。
    asm volatile("tcgen05.fence::before_thread_sync;":::"memory");
    __syncthreads();
  }
  if constexpr(Update)
    for(int i=tid;i<Elements;i+=Threads) out[id*Elements+i]=s.state[i];
  // dealloc是跨warp读之后的资源回收；所有warp先完成各自的读与fence。
  asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
  if(tid==0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(bar):"memory");
  if(warp==0)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,128;"::"r"(tmem):"memory");
}

struct Device {
  std::uint16_t *a=nullptr,*bt=nullptr,*s=nullptr,*out=nullptr;
  float *g=nullptr,*product=nullptr;
  Inputs inputs() const { return {a,bt,s,g}; }
};
template<class T> void allocate(T*& ptr,std::size_t count) { CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&ptr),count*sizeof(T))); }
struct Reference {
  std::vector<float> product;
  std::vector<std::uint16_t> step1,step32;
};
Reference make_input(Device& d,int blocks) {
  std::vector<std::uint16_t> a(blocks*M*K),bt(blocks*N*K),s(blocks*Elements);
  std::vector<float> g(blocks*M);
  Reference ref; ref.product.resize(blocks*Elements); ref.step1.resize(blocks*Elements); ref.step32.resize(blocks*Elements);
  for(int id=0;id<blocks;++id) {
    for(int row=0;row<M;++row) {
      for(int k=0;k<K;++k) {
        a[id*M*K+packed(row,k)]=to_bf16(fixture(0,id,row,k));
        bt[id*N*K+packed(row,k)]=to_bf16(fixture(1,id,row,k));
      }
      g[id*M+row]=gate(id,row);
      for(int col=0;col<N;++col) {
        int at=id*Elements+state_at(row,col);
        s[at]=to_bf16(fixture(2,id,row,col));
        float p=product_reference(id,row,col); ref.product[at]=p;
        ref.step1[at]=update_reference(id,row,col,1,p);
        ref.step32[at]=update_reference(id,row,col,32,p);
      }
    }
  }
  allocate(d.a,a.size()); allocate(d.bt,bt.size()); allocate(d.s,s.size());
  allocate(d.g,g.size()); allocate(d.out,s.size()); allocate(d.product,s.size());
  CUDA_OK(cudaMemcpy(d.a,a.data(),a.size()*2,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d.bt,bt.data(),bt.size()*2,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d.s,s.data(),s.size()*2,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d.g,g.data(),g.size()*4,cudaMemcpyHostToDevice));
  return ref;
}

// 先比较未压成BF16的完整FP32 GEMM；再验证每一步都BF16舍入的状态更新。
void validate_product(Device& d,const Reference& ref,int blocks,bool tc,cudaStream_t stream) {
  CUDA_OK(cudaMemsetAsync(d.product,0xff,ref.product.size()*4,stream));
  if(tc) phase6_tcgen<false><<<blocks,Threads,sizeof(Shared),stream>>>(d.inputs(),nullptr,d.product,1);
  else phase6_sm80<false><<<blocks,Threads,sizeof(Shared),stream>>>(d.inputs(),nullptr,d.product,1);
  CUDA_OK(cudaGetLastError());
  std::vector<float> output(ref.product.size());
  CUDA_OK(cudaMemcpyAsync(output.data(),d.product,output.size()*4,cudaMemcpyDeviceToHost,stream));
  CUDA_OK(cudaStreamSynchronize(stream));
  std::size_t wrong=0,nonfinite=0,first=output.size(); double max_abs=0;
  for(std::size_t i=0;i<output.size();++i) {
    if(!std::isfinite(output[i])) ++nonfinite;
    if(output[i]!=ref.product[i]) { ++wrong; first=std::min(first,i); }
    if(std::isfinite(output[i])) max_abs=std::max(max_abs,std::abs(double(output[i])-ref.product[i]));
  }
  std::printf("GEMM_CHECK,impl=%s,matrices=%d,elements=%zu,wrong=%zu,nonfinite=%zu,max_abs=%.9g,status=%s\n",
    tc?"tcgen05":"sm80",blocks,output.size(),wrong,nonfinite,max_abs,wrong?"FAIL":"PASS");
  if(wrong) {
    std::printf("FIRST_MISMATCH,index=%zu,observed=%.9g,expected=%.9g\n",first,output[first],ref.product[first]);
    throw std::runtime_error("exact FP32 GEMM check failed; timing skipped");
  }
}
void check_state(Device& d,const Reference& ref,int blocks,bool tc,int steps,const char* phase,cudaStream_t stream) {
  const auto& expected=steps==1?ref.step1:ref.step32;
  std::vector<std::uint16_t> output(expected.size());
  CUDA_OK(cudaMemcpyAsync(output.data(),d.out,output.size()*2,cudaMemcpyDeviceToHost,stream));
  CUDA_OK(cudaStreamSynchronize(stream));
  std::size_t wrong=0,nonfinite=0,first=output.size(); double max_abs=0;
  for(std::size_t i=0;i<output.size();++i) {
    float v=from_bf16(output[i]);
    if(!std::isfinite(v)) ++nonfinite;
    if(output[i]!=expected[i]) { ++wrong; first=std::min(first,i); }
    if(std::isfinite(v)) max_abs=std::max(max_abs,std::abs(double(v)-from_bf16(expected[i])));
  }
  std::printf("STATE_CHECK,impl=%s,steps=%d,phase=%s,matrices=%d,elements=%zu,bit_mismatches=%zu,nonfinite=%zu,max_abs=%.9g,status=%s\n",
    tc?"tcgen05":"sm80",steps,phase,blocks,output.size(),wrong,nonfinite,max_abs,wrong?"FAIL":"PASS");
  if(wrong) {
    std::printf("FIRST_MISMATCH,index=%zu,observed=%.9g,expected=%.9g\n",first,from_bf16(output[first]),from_bf16(expected[first]));
    throw std::runtime_error("BF16 stepwise state check failed; timing skipped");
  }
}
void launch(Device& d,int blocks,bool tc,int steps,cudaStream_t stream) {
  if(tc) phase6_tcgen<true><<<blocks,Threads,sizeof(Shared),stream>>>(d.inputs(),d.out,nullptr,steps);
  else phase6_sm80<true><<<blocks,Threads,sizeof(Shared),stream>>>(d.inputs(),d.out,nullptr,steps);
  CUDA_OK(cudaGetLastError());
}
struct Case {
  bool tc=false; int steps=1;
  cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
  std::vector<double> times;
};
template<class Kernel> void resource(Kernel fn,const char* name) {
  cudaFuncAttributes a{}; CUDA_OK(cudaFuncSetAttribute(fn,cudaFuncAttributePreferredSharedMemoryCarveout,100));
  CUDA_OK(cudaFuncGetAttributes(&a,fn)); int blocks=0;
  CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,Threads,sizeof(Shared)));
  std::printf("RESOURCE,impl=%s,threads=%d,registers=%d,static_smem=%zu,dynamic_smem=%zu,local_bytes=%zu,occupancy_API_blocks=%d,tmem_columns=%d\n",
    name,Threads,a.numRegs,a.sharedSizeBytes,sizeof(Shared),a.localSizeBytes,blocks,std::string(name)=="tcgen05"?128:0);
}
int main(int argc,char** argv) {
  std::setvbuf(stdout,nullptr,_IOLBF,0);
  try {
    if(argc!=5) throw std::runtime_error("expected blocks graph_iterations rounds warmup_graphs");
    int blocks=std::stoi(argv[1]),iterations=std::stoi(argv[2]),rounds=std::stoi(argv[3]),warmup=std::stoi(argv[4]);
    if(blocks<1||blocks>4096||iterations<1||iterations>100||rounds<4||rounds>100||rounds%4||warmup<1||warmup>100)
      throw std::runtime_error("parameters outside supported bounds; rounds must be divisible by 4");
    int device,driver,runtime; CUDA_OK(cudaGetDevice(&device)); cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop,device)); CUDA_OK(cudaDriverGetVersion(&driver)); CUDA_OK(cudaRuntimeGetVersion(&runtime));
    std::printf("device=%s,capability=%d.%d,SMs=%d,driver=%d,runtime=%d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount,driver,runtime);
    std::puts("evidence=REAL_PHASE6_SHAPE_COMPARISON; CHUNK=16; not_full_FlashKDA; not_instruction_latency");
    std::puts("equation=S_next=BF16_RN(FP32_FMA(S,g,A_times_B)); A_B_reused_across_steps; no_prior_phase_U_materialization");
    std::puts("scope=common_packed_SMEM_inputs_column_major_state; global_staging_and_final_store_included; TMEM_alloc_free_included_once_per_CTA");
    std::puts("clocks=unmodified; buffers=reused; cache_flush=no; occupancy_API_is_not_achieved_or_TMEM_residency");
    std::printf("CONFIG,M=%d,N=%d,K=%d,blocks=%d,steps=1_and_32,graph_iterations=%d,rounds=%d,warmup_graphs=%d,idesc=0x%08x\n",
      M,N,K,blocks,iterations,rounds,warmup,IDesc);
    Device d; Reference ref=make_input(d,blocks);
    cudaStream_t stream; CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    resource(phase6_sm80<true>,"sm80"); resource(phase6_tcgen<true>,"tcgen05");
    validate_product(d,ref,blocks,false,stream); validate_product(d,ref,blocks,true,stream);
    std::vector<Case> cases;
    for(int steps : {1,32}) for(bool tc : {false,true}) {
      Case c; c.tc=tc; c.steps=steps;
      CUDA_OK(cudaMemsetAsync(d.out,0xff,ref.step1.size()*2,stream));
      launch(d,blocks,tc,steps,stream); check_state(d,ref,blocks,tc,steps,"direct",stream);
      CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
      for(int i=0;i<iterations;++i) launch(d,blocks,tc,steps,stream);
      CUDA_OK(cudaStreamEndCapture(stream,&c.graph));
      std::size_t nodes=0; CUDA_OK(cudaGraphGetNodes(c.graph,nullptr,&nodes));
      if(nodes!=std::size_t(iterations)) throw std::runtime_error("Graph node count mismatch");
      CUDA_OK(cudaGraphInstantiateWithFlags(&c.exec,c.graph,0));
      CUDA_OK(cudaMemsetAsync(d.out,0xff,ref.step1.size()*2,stream));
      for(int i=0;i<warmup;++i) CUDA_OK(cudaGraphLaunch(c.exec,stream));
      check_state(d,ref,blocks,tc,steps,"graph_warmup",stream);
      std::printf("GRAPH,impl=%s,steps=%d,kernel_nodes=%zu\n",tc?"tcgen05":"sm80",steps,nodes);
      cases.push_back(c);
    }
    std::puts("all_prechecks=PASS; timing_starts_after_this_line");
    cudaEvent_t start,stop; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    for(int r=0;r<rounds;++r) for(int slot=0;slot<4;++slot) {
      auto& c=cases[(r+slot)%4]; float ms=0;
      CUDA_OK(cudaEventRecord(start,stream)); CUDA_OK(cudaGraphLaunch(c.exec,stream));
      CUDA_OK(cudaEventRecord(stop,stream)); CUDA_OK(cudaEventSynchronize(stop));
      CUDA_OK(cudaEventElapsedTime(&ms,start,stop));
      double batch=double(ms)/iterations;
      if(!std::isfinite(batch)||batch<=0) throw std::runtime_error("invalid timing");
      c.times.push_back(batch);
      std::printf("RAW_TIME,impl=%s,steps=%d,round=%d,order_slot=%d,graph_ms=%.9g,batch_ms=%.9g\n",c.tc?"tcgen05":"sm80",c.steps,r,slot,double(ms),batch);
    }
    // 仅对同一blocks/steps定义 speedup=SM80/tcgen；除以steps不是纯指令latency。
    for(int steps : {1,32}) {
      double medians[2]={};
      for(auto& c:cases) if(c.steps==steps) {
        auto values=c.times; std::sort(values.begin(),values.end());
        double median=0.5*(values[values.size()/2-1]+values[values.size()/2]); medians[c.tc]=median;
        double mean=0; for(double t:values) mean+=t; mean/=values.size();
        double flops=double(blocks)*steps*524288;
        std::printf("SUMMARY,impl=%s,steps=%d,blocks=%d,median_batch_ms=%.9g,mean_batch_ms=%.9g,min_batch_ms=%.9g,max_batch_ms=%.9g,normalized_update_batch_us=%.9g,useful_MMA_TFLOPs=%.9g\n",
          c.tc?"tcgen05":"sm80",steps,blocks,median,mean,values.front(),values.back(),median*1000/steps,flops/(median*1e9));
      }
      std::printf("COMPARISON,steps=%d,blocks=%d,speedup_sm80_over_tcgen05=%.9g\n",steps,blocks,medians[0]/medians[1]);
    }
    // 再次各运行一次并检查完整输出，防止只验证最初的正确性。
    for(auto& c:cases) {
      CUDA_OK(cudaMemsetAsync(d.out,0xff,ref.step1.size()*2,stream));
      CUDA_OK(cudaGraphLaunch(c.exec,stream));
      check_state(d,ref,blocks,c.tc,c.steps,"after_timing",stream);
    }
    std::puts("timing_completed=yes; all_postchecks=PASS; speedups_apply_only_to_this_phase6_like_implementation");
    for(auto& c:cases) { CUDA_OK(cudaGraphExecDestroy(c.exec)); CUDA_OK(cudaGraphDestroy(c.graph)); }
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop)); CUDA_OK(cudaStreamDestroy(stream));
    CUDA_OK(cudaFree(d.a)); CUDA_OK(cudaFree(d.bt)); CUDA_OK(cudaFree(d.s));
    CUDA_OK(cudaFree(d.out)); CUDA_OK(cudaFree(d.g)); CUDA_OK(cudaFree(d.product));
    return 0;
  } catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 2; }
}
#endif
'''

# 【Python运行与证据】只负责编译/执行/日志，不参与GPU计时区间。
def run(args):
    work = args.work_dir.resolve()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    suffix = 'offline_task2_phase6' if args.offline_check else 'task2_phase6'
    raw = work / 'logs/raw'
    build = work / 'build' / f'{run_id}_{suffix}'
    raw.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=False)
    log_path = raw / f'{run_id}_{suffix}.log'
    before = raw / f'{run_id}_{suffix}_gpu_before.csv'
    after = raw / f'{run_id}_{suffix}_gpu_after.csv'
    source, binary = build / 'phase6.cu', build / 'phase6'
    sass = raw / f'{run_id}_{suffix}_sass.txt'
    artifacts = [Path(__file__).resolve(), source, binary, sass, before, after, log_path]
    status, attempted_gpu = 1, False
    with log_path.open('x') as log:
        def emit(value):
            print(value, flush=True)
            log.write(str(value) + '\n')
            log.flush()

        def command(argv, timeout, output_file=None):
            # 以参数列表调用，路径中有空格也不会变成shell命令。
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

        emit('purpose=Task2_discussion2_Phase6_SM80_vs_tcgen05; not_FlashKDA_benchmark')
        emit(f'mode={"OFFLINE_CPU_ONLY" if args.offline_check else "CUDA_REQUESTED_NOT_YET_COMPLETED"}')
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
                         '-ffp-contract=off', '-DPHASE6_OFFLINE', str(source), '-o', str(binary)], 60)
                command([str(binary)], 30)
            else:
                if not os.environ.get('SLURM_JOB_ID'):
                    raise RuntimeError('Run inside the existing Slurm GPU allocation.')
                if not args.nvcc.is_file() or not os.access(args.nvcc, os.X_OK):
                    raise RuntimeError(f'nvcc not executable: {args.nvcc}')
                if shutil.which('nvidia-smi') is None:
                    raise RuntimeError('nvidia-smi unavailable')
                command([str(args.nvcc), '--version'], 20)
                # 明确指定虚拟架构与机器码目标：本实验只构建B300的sm_103a。
                # -arch=sm_103a简写还会加入通用compute_103 PTX；其中的tcgen05会被拒绝。
                # 保留a后缀，避免在通用sm_103目标下校验本文件的内联PTX。
                command([str(args.nvcc), '-std=c++17', '-O3', '-gencode=arch=compute_103a,code=sm_103a', '--fmad=false',
                         '-Xcompiler=-ffp-contract=off', '-Xptxas=-v', str(source), '-o', str(binary)], 240)
                cuobjdump = args.nvcc.parent / 'cuobjdump'
                if not cuobjdump.is_file():
                    raise RuntimeError('cuobjdump unavailable; this comparison requires both opcode families')
                listing = command([str(cuobjdump), '--dump-sass', str(binary)], 60, sass)
                # 按完整opcode及kernel归属统计，避免把UTCHMMA误算为HMMA后缀。
                functions, current = {}, None
                for line in listing.splitlines():
                    match = re.search(r'\bFunction\s*:\s*(\S+)', line)
                    if match:
                        current = match.group(1)
                        functions.setdefault(current, Counter())
                    # 先跳过P/UP谓词（含否定与PT/UPT），再读取完整opcode。
                    # 实机UTCHMMA带@UP1；只识别@P会漏掉它，误报指令族缺失。
                    op = re.search(r'/\*[0-9a-fA-F]+\*/\s*(?:@!?U?P(?:\d+|T)\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)', line)
                    if current and op:
                        functions[current][op.group(1)] += 1
                family_totals = Counter()
                for name, counts in sorted(functions.items()):
                    if 'phase6_sm80' not in name and 'phase6_tcgen' not in name:
                        continue
                    selected = {op: count for op, count in sorted(counts.items())
                                if op.split('.')[0] in ('HMMA', 'UTCHMMA', 'LDSM')}
                    emit(f'SASS_KERNEL,name={name},static_variants={selected}')
                    for op, count in counts.items():
                        if 'phase6_sm80' in name and op.split('.')[0] == 'HMMA':
                            family_totals['sm80_HMMA'] += count
                        if 'phase6_tcgen' in name and op.split('.')[0] == 'UTCHMMA':
                            family_totals['tcgen_UTCHMMA'] += count
                emit('SASS_FAMILY_TOTALS=' + repr(dict(family_totals)))
                emit('SASS_counts_are_static_and_include_correctness_kernels; not_dynamic_instruction_counts')
                if not family_totals['sm80_HMMA'] or not family_totals['tcgen_UTCHMMA']:
                    raise RuntimeError('expected instruction family missing; keep SASS for diagnosis, skip timing')
                snapshot(before)
                attempted_gpu = True
                command([str(binary), str(args.blocks), str(args.graph_iterations),
                         str(args.rounds), str(args.warmup_graphs)], 180)
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
            emit(f'probe_exit={status}')
            emit('scope=independent_Phase6_shape_and_epilogue; full_KDA_and_prior_U_materialization_not_tested')
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
    parser.add_argument('--offline-check', action='store_true', help='仅核对CPU布局/descriptor/参考，绝不启动GPU')
    parser.add_argument('--work-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'runs')
    parser.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    parser.add_argument('--blocks', type=int, default=96, help='独立状态矩阵数；默认对应fixed基线的96个CTA')
    parser.add_argument('--graph-iterations', type=int, default=5, help='每个Graph内的完整grid数')
    parser.add_argument('--rounds', type=int, default=12, help='每种impl/steps的采样数；须为4的倍数以平衡顺序')
    parser.add_argument('--warmup-graphs', type=int, default=3, help='每种case在计时前预热的Graph数')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
