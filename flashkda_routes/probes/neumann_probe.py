#!/usr/bin/env python3
"""Task 2 讨论点 1：验证 Neumann 求逆的项数与低精度中间量。

要回答的问题：C 从 16 增至 32/64 后，补足级数是否就能得到准确的逆？
默认模式：在 Slurm GPU 会话中，编译并执行独立 CUDA/cuBLAS 数值实验。
--offline-check：仅编译、运行 CPU 参考和代数对照，不使用 GPU。
两种模式均不执行 FlashKDA 本体，也不测耗时。

建议阅读顺序：
1. fixture：理解五种输入为什么这样构造。
2. forward_inverse / measure：理解参考答案与误差指标。
3. CUDA main 中的 p=2,4,8,... 循环：理解实际求逆过程。
4. Python run：仅负责编译、运行、记录日志和哈希，可以最后看。

日志速读：STAGE 看哪一阶段出现非有限值；RESULT 看最终残差/误差。
algebra_controls=PASS 只表示 zero/shift 对照通过，不表示压力输入都准确。
本文件补充中文注释；实测时的原始字节版本保留在 scripts/archive/ 中。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


# 以下字符串是交给 nvcc/g++ 的完整源文件；Python 不直接执行其中的 C++。
# NEUMANN_OFFLINE_CHECK 把同一文件分成 CPU 对照分支和真实 CUDA 分支。
CUDA_SOURCE = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef NEUMANN_OFFLINE_CHECK
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#endif

// 【数据约定】host 矩阵用一维数组按行存储，(i,j) 对应下标 i*n+j。
// Matrix 中的 double 是 CPU FP64；GPU 运算缓冲区另用 __half。
using Matrix = std::vector<double>;
const char* fixtures[] = {"zero", "shift_one", "aligned_b0p5_g0",
                          "aligned_b1_g0", "random_b0p5_g0p1"};

// 【1. 输入】输出 n×n 严格下三角矩阵 L，对角线和上三角均为 0。
// zero：恒等逆对照；shift_one：验证截断项数的精确代数对照。
// aligned：同方向 keys 的相干压力输入；random：固定种子的较弱相关输入。
// KDA 形式的直接模型：L_ij=beta*dot(k_i,k_j)*exp(-alpha*(i-j)), i>j。
// 这里绕过原 kernel 的 gate/Gram 计算；GPU 分支会先将 L 量化到 FP16。
Matrix fixture(int n, const std::string& name) {
  Matrix l(n*n, 0.0);
  if (name == "zero") return l;
  if (name == "shift_one") {
    for (int i=1; i<n; ++i) l[i*n+i-1] = 1.0;
    return l;
  }
  if (name == "aligned_b0p5_g0" || name == "aligned_b1_g0") {
    double beta = name == "aligned_b1_g0" ? 1.0 : 0.5;
    for (int i=0; i<n; ++i) for (int j=0; j<i; ++j) l[i*n+j] = beta;
    return l;
  }
  if (name != "random_b0p5_g0p1") throw std::runtime_error("unknown fixture");
  // 固定 uint32 线性同余生成器，使不同 C 共享同一个 keys 前缀。
  // 每个 key 有 128 个分量，随后单位化，避免长度差异干扰内积。
  std::uint32_t seed = 20260910u;
  Matrix keys(n*128);
  for (int i=0; i<n; ++i) {
    double norm2 = 0.0;
    for (int d=0; d<128; ++d) {
      seed = 1664525u*seed + 1013904223u;
      double x = static_cast<double>(seed >> 8) / 16777216.0 - 0.5;
      keys[i*128+d] = x;
      norm2 += x*x;
    }
    for (int d=0; d<128; ++d) keys[i*128+d] /= std::sqrt(norm2);
  }
  for (int i=0; i<n; ++i) for (int j=0; j<i; ++j) {
    double dot = 0.0;
    for (int d=0; d<128; ++d) dot += keys[i*128+d]*keys[j*128+d];
    l[i*n+j] = 0.5 * dot * std::exp(-0.1*(i-j));
  }
  return l;
}

// 【2. 参考答案】用 FP64 前代法解 (I+L)R=I；输入必须严格下三角。
// 每一列是一个线性方程组。它不使用 Neumann 级数，避免参考与被测算法同错。
Matrix forward_inverse(const Matrix& l, int n) {
  Matrix r(n*n, 0.0);
  // 因 A=I+L 的对角线为 1，R_ij=delta_ij-sum_{k<i} L_ik*R_kj。
  for (int j=0; j<n; ++j) for (int i=0; i<n; ++i) {
    double sum = 0.0;
    for (int k=0; k<i; ++k) sum += l[i*n+k]*r[k*n+j];
    r[i*n+j] = (i==j ? 1.0 : 0.0) - sum;
  }
  return r;
}

// 矩阵无穷范数：先求每一行的绝对值之和，再取最大值；不是元素最大值。
double inf_norm(const Matrix& a, int n) {
  double result = 0.0;
  for (int i=0; i<n; ++i) {
    double row = 0.0;
    for (int j=0; j<n; ++j) row += std::abs(a[i*n+j]);
    result = std::max(result, row);
  }
  return result;
}

// 【3. 指标】nonfinite 统计 NaN/Inf；max_finite 只描述剩余的有限元素。
// residual_inf=||(I+L)R-I||_inf；error_max=max|R-R_ref|。
// scaled_residual 与 error_relative_inf 是两种不同的归一化，不能混用。
struct Metrics {
  int nonfinite = 0;
  double max_finite = 0.0, residual_inf = 0.0, scaled_residual = 0.0;
  double error_max = 0.0, error_relative_inf = 0.0;
};

// 输入：同一份 L、待评估的 R、独立参考 ref；所有残差都在 CPU FP64 中计算。
// 先检查非有限值，再算范数；防止 max 操作忽略 NaN 而报出看似正常的小误差。
Metrics measure(const Matrix& l, const Matrix& r, const Matrix& ref, int n) {
  Metrics m;
  for (double v : r) {
    if (!std::isfinite(v)) ++m.nonfinite;
    else m.max_finite = std::max(m.max_finite, std::abs(v));
  }
  if (m.nonfinite) {
    m.residual_inf = m.scaled_residual = m.error_max = m.error_relative_inf =
      std::numeric_limits<double>::quiet_NaN();
    return m; // 一旦有 NaN/Inf，误差列明确填 NaN，不把有限子集当成完整结果。
  }
  Matrix residual(n*n), error(n*n), a=l;
  for (int i=0; i<n; ++i) a[i*n+i] += 1.0;
  for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) {
    double ar = r[i*n+j];
    for (int k=0; k<i; ++k) ar += l[i*n+k]*r[k*n+j];
    residual[i*n+j] = ar - (i==j ? 1.0 : 0.0);
    error[i*n+j] = r[i*n+j]-ref[i*n+j];
    m.error_max = std::max(m.error_max, std::abs(error[i*n+j]));
  }
  m.residual_inf = inf_norm(residual,n);
  m.scaled_residual = m.residual_inf / (inf_norm(a,n)*inf_norm(r,n)+1.0);
  m.error_relative_inf = inf_norm(error,n) / inf_norm(ref,n);
  return m;
}

// 【4A. 离线分支】只供 --offline-check 使用，不模拟 FP16 或 GPU 指令。
#ifdef NEUMANN_OFFLINE_CHECK
// 朴素 FP64 矩阵乘法，仅为小尺寸代数对照服务，不用来测性能。
Matrix multiply(const Matrix& a, const Matrix& b, int n) {
  Matrix c(n*n, 0.0);
  for (int i=0; i<n; ++i) for (int j=0; j<n; ++j)
    for (int k=0; k<n; ++k) c[i*n+j] += a[i*n+k]*b[k*n+j];
  return c;
}

// cutoff 是多项式的项数：R_cutoff=I-L+...+(-L)^(cutoff-1)。
// 本实验 cutoff 为 16/32/64；平方倍增把项数逐次从 2 扩到 4、8、16……。
Matrix neumann_double(const Matrix& l, int n, int cutoff) {
  Matrix r=l, power=l;
  for (int i=0; i<n; ++i) for (int j=0; j<n; ++j)
    r[i*n+j] = (i==j ? 1.0 : 0.0)-l[i*n+j];
  for (int p=2; p<cutoff; p*=2) {
    power = multiply(power,power,n);
    Matrix temp=multiply(r,power,n);
    for (int i=0; i<n*n; ++i) r[i] += temp[i];
  }
  return r;
}

// 计算组合数的纸面预算；double 存储，大整数输出不是精确整数验证。
double choose(int n, int k) {
  double result=1.0;
  for (int i=1; i<=k; ++i) result *= double(n-k+i)/i;
  return result;
}

// CPU 入口：检查前代参考、相干矩阵的闭式逆，以及 shift 的截断残差。
// 输出 PAPER_ALIGNED_B1 的组合公式预测，不把预测标成 GPU 结果。
int main() {
  std::puts("evidence=OFFLINE_CPP_REFERENCE_AND_ALGEBRA; gpu_executed=no; no_FP16_emulation");
  bool ok=true;
  for (int n : {16,32,64}) {
    for (const char* name : fixtures) {
      Matrix l=fixture(n,name), ref=forward_inverse(l,n);
      Metrics rm=measure(l,ref,ref,n);
      bool valid=std::isfinite(rm.residual_inf) && rm.residual_inf<1e-12;
      double closed_error=0.0;
      if (std::string(name).find("aligned_")==0) {
        double beta = std::string(name)=="aligned_b1_g0" ? 1.0 : 0.5;
        for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) {
          double expected=i==j ? 1.0 : (i>j ? -beta*std::pow(1.0-beta,i-j-1) : 0.0);
          closed_error=std::max(closed_error,std::abs(ref[i*n+j]-expected));
        }
        valid=valid && closed_error<1e-12;
      }
      std::printf("REFERENCE,%s,%d,residual_inf=%.9g,closed_form_error_max=%.9g,valid=%s\n",
                  name,n,rm.residual_inf,closed_error,valid?"yes":"no");
      ok=ok && valid;
    }
    Matrix l=fixture(n,"shift_one"), ref=forward_inverse(l,n);
    Metrics full=measure(l,neumann_double(l,n,n),ref,n);
    Metrics short_m=measure(l,neumann_double(l,n,16),ref,n);
    double expected=n==16 ? 0.0 : 1.0;
    bool valid=full.residual_inf==0.0 && full.error_max==0.0 && short_m.residual_inf==expected;
    ok=ok && valid;
    std::printf("SHIFT_CONTROL,C=%d,full_residual_inf=%.9g,truncated16_residual_inf=%.9g,expected_truncated=%.9g,valid=%s\n",
                n,full.residual_inf,short_m.residual_inf,expected,valid?"yes":"no");
    for (int p=2; p<n; p*=2)
      std::printf("PAPER_ALIGNED_B1,C=%d,power=%d,max_entry=%.12g,fp16_max=65504\n",n,p,choose(n-2,p-1));
  }
  std::printf("offline_controls=%s; CUDA_compiled=no; CUDA_executed=no\n",ok?"PASS":"FAIL");
  return ok ? 0 : 3;
}
#else

// 【4B. 实机分支】以下调用 CUDA/cuBLAS；任何 API 错误都输出位置并返回非零。

#define CUDA_CHECK(expr) do { cudaError_t e=(expr); if(e!=cudaSuccess) \
  throw std::runtime_error(std::string("CUDA line=")+std::to_string(__LINE__)+" "+cudaGetErrorString(e)); } while(0)
#define BLAS_CHECK(expr) do { cublasStatus_t e=(expr); if(e!=CUBLAS_STATUS_SUCCESS) \
  throw std::runtime_error(std::string("CUBLAS line=")+std::to_string(__LINE__)+" status="+std::to_string(int(e))); } while(0)

// 初始化前两项 R_2=I-L；一线程写一个元素。
__global__ void init_inverse(const __half* l, __half* r, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n*n) r[i]=__float2half_rn((i/n==i%n ? 1.0f : 0.0f)-__half2float(l[i]));
}
// 乘法结果 delta 已是 FP16，再独立做一次 FP16 加法，保留这一舍入阶段。
__global__ void add_half(__half* r, const __half* delta, int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) r[i]=__hadd(r[i],delta[i]);
}
// 评估最终逆转成 BF16 后的误差；这不是 recurrent state 的长期精度实验。
__global__ void to_bfloat16(const __half* r, __nv_bfloat16* out, int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) out[i]=__float2bfloat16_rn(__half2float(r[i]));
}

// 【矩阵乘法包装】计算 c=a*b，输入/输出均为 FP16，c 不与输入复用缓冲区。
// COMPUTE_16F 请求至少 FP16 的内部精度，不能据此断言与原 HMMA 舍入次序一致。
void gemm(cublasHandle_t handle, const __half* a, const __half* b, __half* c, int n) {
  __half one=__float2half_rn(1.0f), zero=__float2half_rn(0.0f);
  // cuBLAS 按列解释存储，因此交换 a/b，以 C^T=B^T*A^T 得到按行存储的 C。
  // alpha=1、beta=0 表示只写 a*b，不累加原有 c。
  BLAS_CHECK(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,
                        b,CUDA_R_16F,n,a,CUDA_R_16F,n,&zero,c,CUDA_R_16F,n,
                        CUBLAS_COMPUTE_16F,CUBLAS_GEMM_DEFAULT));
}

// 把 device FP16 矩阵复制到 CPU，再精确提升到 double；复制会等待计算。
// 这些逐阶段同步用于取得证据，所以本脚本的执行耗时不能作为 GEMM 性能。
Matrix read_half(const __half* data, int n) {
  std::vector<__half> host(n*n);
  CUDA_CHECK(cudaMemcpy(host.data(),data,host.size()*sizeof(__half),cudaMemcpyDeviceToHost));
  Matrix out(host.size());
  for(std::size_t i=0;i<host.size();++i) out[i]=__half2float(host[i]);
  return out;
}

// 输出一行 RESULT。返回值只检查有精确预期的 zero/shift 流程对照。
// cutoff<n 表示尚未完成 n 项级数；压力输入误差大小需结合数据判断。
bool report(const char* name, int n, int cutoff, const char* storage,
            const Matrix& l, const Matrix& r, const Matrix& ref, double ref_residual) {
  Metrics m=measure(l,r,ref,n);
  std::printf("RESULT,%s,%d,%d,%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
      name,n,cutoff,storage,m.nonfinite,m.max_finite,m.residual_inf,m.scaled_residual,
      m.error_max,m.error_relative_inf,ref_residual);
  if(std::string(name)=="zero") return m.nonfinite==0 && m.error_max==0.0 && m.residual_inf==0.0;
  if(std::string(name)=="shift_one") {
    double expected=cutoff<n ? 1.0 : 0.0;
    return m.nonfinite==0 && m.residual_inf==expected && (cutoff<n || m.error_max==0.0);
  }
  return true; // 压力输入记录观测；这里的 true 不代表其求逆精度合格。
}

// CUDA 入口：配置设备和 cuBLAS → 逐个输入求逆 → 回传各阶段 → 检查对照。
int main() {
  try {
    int device=0,driver=0,runtime=0,blas_version=0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    cublasHandle_t handle;
    BLAS_CHECK(cublasCreate(&handle));
    BLAS_CHECK(cublasGetVersion(handle,&blas_version));
    BLAS_CHECK(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
    BLAS_CHECK(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
    BLAS_CHECK(cublasSetStream(handle,0));
    std::puts("evidence=REAL_CUDA_CUBLAS_NEUMANN_NUMERICAL_PROBE");
    std::printf("device=%s capability=%d.%d sm_count=%d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);
    std::printf("cuda_driver=%d cuda_runtime=%d cublas_version=%d\n",driver,runtime,blas_version);
    std::puts("scope=direct_L_matrices; FP16_input_output; CUBLAS_COMPUTE_16F; separate_half_add; final_bf16_RN; FP64_CPU_reference; no_latency");
    std::puts("not_exact_FlashKDA_HMMA_order; no_gate_factorization; no_full_KDA; no_state_accuracy_claim");
    std::puts("fixture_generation=CPU_FP64; L_RN_FP16_once_via_FP32; reference_uses_same_quantized_L; random_LCG_seed=20260910; D=128");
    std::puts("STAGE_columns=tag,fixture,C,power,power_nonfinite,power_max_finite,inverse_nonfinite");
    std::puts("RESULT_columns=tag,fixture,C,cutoff,storage,nonfinite,max_finite,residual_inf,scaled_residual,inverse_error_max,inverse_error_relative_inf,reference_residual_inf");
    bool controls_ok=true;
    // 五种输入 × 三个矩阵尺寸。只量化一次，CPU 参考也用量化后的同一个 L。
    for(int n : {16,32,64}) for(const char* name : fixtures) {
      Matrix l=fixture(n,name);
      std::vector<__half> input(n*n);
      for(int i=0;i<n*n;++i) {
        input[i]=__float2half_rn(static_cast<float>(l[i]));
        l[i]=__half2float(input[i]);
      }
      Matrix ref=forward_inverse(l,n);
      double ref_residual=measure(l,ref,ref,n).residual_inf;
      if(!std::isfinite(ref_residual) || ref_residual>1e-12)
        throw std::runtime_error("FP64 reference residual is unexpectedly large");
      // power=L^p；next 是平方的输出；r 是当前级数；temp 是增量 r*power。
      // 分开输入与输出缓冲区，避免 GEMM 原地覆盖仍需读取的数据。
      __half *power=nullptr,*next=nullptr,*r=nullptr,*temp=nullptr;
      __nv_bfloat16* bf=nullptr;
      std::size_t bytes=n*n*sizeof(__half);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&power),bytes));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&next),bytes));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&r),bytes));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp),bytes));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&bf),n*n*sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMemcpy(power,input.data(),bytes,cudaMemcpyHostToDevice));
      init_inverse<<<(n*n+127)/128,128>>>(power,r,n);
      CUDA_CHECK(cudaGetLastError());
      // 本循环是核心：在 p 阶段，先得到 L^p，再把 R_p 扩为 R_(2p)。
      // 即使 L^p 有限，temp 或新的 r 仍可能溢出；输出分别统计 power 与 r。
      // temp 没有单独回传，因此只能定位到该更新阶段，不能区分具体哪条操作先坏。
      for(int p=2;p<n;p*=2) {
        gemm(handle,power,power,next,n);
        std::swap(power,next);
        gemm(handle,r,power,temp,n);
        add_half<<<(n*n+127)/128,128>>>(r,temp,n*n);
        CUDA_CHECK(cudaGetLastError());
        Matrix host_power=read_half(power,n), host_r=read_half(r,n);
        int power_bad=0,r_bad=0;
        double max_power=0.0;
        for(double v:host_power) {
          if(!std::isfinite(v)) ++power_bad;
          else max_power=std::max(max_power,std::abs(v));
        }
        for(double v:host_r) if(!std::isfinite(v)) ++r_bad;
        std::printf("STAGE,%s,%d,%d,%d,%.9g,%d\n",name,n,p,power_bad,max_power,r_bad);
        // p=8 时记录 R_16；最后一阶段再记录完整 R_n。n=16 时只记录一次。
        if(p==8 || p==n/2) {
          int cutoff=2*p;
          controls_ok=report(name,n,cutoff,"fp16",l,host_r,ref,ref_residual) && controls_ok;
          // BF16 只是输出对照；转换结果不送回 r，不影响后面的 FP16 级数。
          to_bfloat16<<<(n*n+127)/128,128>>>(r,bf,n*n);
          CUDA_CHECK(cudaGetLastError());
          std::vector<__nv_bfloat16> host_bf(n*n);
          CUDA_CHECK(cudaMemcpy(host_bf.data(),bf,n*n*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
          Matrix rb(n*n);
          for(int i=0;i<n*n;++i) rb[i]=__bfloat162float(host_bf[i]);
          controls_ok=report(name,n,cutoff,"bf16",l,rb,ref,ref_residual) && controls_ok;
        }
      }
      CUDA_CHECK(cudaFree(power)); CUDA_CHECK(cudaFree(next));
      CUDA_CHECK(cudaFree(r)); CUDA_CHECK(cudaFree(temp)); CUDA_CHECK(cudaFree(bf));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    BLAS_CHECK(cublasDestroy(handle));
    std::printf("algebra_controls=%s; stress_errors_are_observations_not_accuracy_passes\n",controls_ok?"PASS":"FAIL");
    return controls_ok ? 0 : 3;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"ERROR: %s\n",e.what());
    return 2;
  }
}
#endif
'''


# 【5. Python 运行包装】不参与矩阵运算，负责环境、编译、进程和证据落盘。
def run(args):
    work = args.work_dir.resolve()
    # 微秒 UTC 路径 + 排他创建：每次独立记录，保留以前的成功或失败证据。
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    suffix = 'offline_task2_neumann' if args.offline_check else 'task2_neumann'
    raw_dir = work / 'logs/raw'
    build = work / 'build' / f'{run_id}_{suffix}'
    raw_dir.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=False)
    log_path = raw_dir / f'{run_id}_{suffix}.log'
    before = raw_dir / f'{run_id}_{suffix}_gpu_before.csv'
    after = raw_dir / f'{run_id}_{suffix}_gpu_after.csv'
    source = build / 'neumann_probe.cu'
    binary = build / 'neumann_probe'
    artifacts = [Path(__file__).resolve(), source, binary, before, after, log_path]
    status, attempted_gpu = 1, False
    with log_path.open('x') as log:
        # 同一内容同时写到屏幕和原始日志；后续直接粘贴，不手工改写数值。
        def emit(value):
            print(value, flush=True)
            log.write(value + '\n')
            log.flush()

        # 参数按列表交给子进程；合并 stdout/stderr，非零退出码转成可记录的错误。
        def command(argv, timeout):
            emit('command=' + shlex.join([str(part) for part in argv]))
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=timeout, check=False)
            if proc.stdout:
                emit(proc.stdout.rstrip('\n'))
            emit(f'command_exit={proc.returncode}')
            if proc.returncode:
                raise RuntimeError(f'command failed with exit {proc.returncode}')

        # 仅记录运行前后的 GPU 进程列表，不证明整个运行期间没有外部干扰。
        def snapshot(path):
            proc = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                   '--format=csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=20, check=False)
            path.write_text(proc.stdout)
            emit('gpu_snapshot=' + str(path))
            emit(proc.stdout.rstrip('\n'))
            if proc.returncode:
                raise RuntimeError(f'nvidia-smi failed with exit {proc.returncode}')

        # REAL_CUDA_PENDING 是启动时状态；以之后的设备输出、退出码判定执行情况。
        emit('purpose=Task2_discussion1_Neumann_numerics; not_FlashKDA_benchmark')
        emit(f'mode={"OFFLINE_CPU_ONLY" if args.offline_check else "REAL_CUDA_PENDING"}')
        emit(f'utc_run_id={run_id}')
        emit(f'python={sys.executable}')
        emit(f'script={Path(__file__).resolve()}')
        emit(f'slurm_job_id={os.environ.get("SLURM_JOB_ID", "missing")}')
        emit(f'raw_log={log_path}')
        try:
            source.write_text(CUDA_SOURCE)
            # 离线只编译 CPU 分支；默认模式要求已有 Slurm allocation，不自行申请 GPU。
            if args.offline_check:
                compiler = shutil.which('g++')
                if compiler is None:
                    raise RuntimeError('g++ unavailable for offline reference controls')
                command([compiler, '--version'], 20)
                command([compiler, '-x', 'c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                         '-pedantic', '-ffp-contract=off', '-DNEUMANN_OFFLINE_CHECK',
                         str(source), '-o', str(binary)], 60)
            else:
                if not os.environ.get('SLURM_JOB_ID'):
                    raise RuntimeError('Run inside the existing Slurm GPU allocation.')
                if not args.nvcc.is_file() or not os.access(args.nvcc, os.X_OK):
                    raise RuntimeError(f'nvcc not executable: {args.nvcc}')
                if shutil.which('nvidia-smi') is None:
                    raise RuntimeError('nvidia-smi unavailable')
                # 从实际 nvcc 所属 toolkit 找开发头和库；缺少依赖就记录错误，不猜路径。
                cuda_root = args.nvcc.resolve().parents[1]
                include_dir = next((p for p in (cuda_root/'include', cuda_root/'targets/x86_64-linux/include')
                                    if (p/'cublas_v2.h').is_file()), None)
                library_dir = next((p for p in (cuda_root/'lib64', cuda_root/'targets/x86_64-linux/lib')
                                    if (p/'libcublas.so').is_file()), None)
                if include_dir is None or library_dir is None:
                    raise RuntimeError(f'cuBLAS development header/library unavailable under {cuda_root}')
                emit(f'cublas_include_dir={include_dir} cublas_library_dir={library_dir}')
                command([str(args.nvcc), '--version'], 20)
                command([str(args.nvcc), '-std=c++17', '-O2', '-arch=sm_103a', '--fmad=false',
                         '-Xcompiler=-ffp-contract=off', '-I', str(include_dir), str(source),
                         '-L', str(library_dir), '-Xlinker', '-rpath', '-Xlinker', str(library_dir),
                         '-lcublas', '-o', str(binary)], 240)
                snapshot(before)
                attempted_gpu = True
            command([str(binary)], 60)
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
            emit('scope=Neumann_numerical_probe_only; no_latency; FlashKDA_C32_C64_not_implemented')
    # 日志关闭后再算哈希，保证清单对应最终字节；注释改变也会改变源文件哈希。
    manifest = log_path.with_name(log_path.name + '.sha256')
    with manifest.open('x') as sha:
        for path in artifacts:
            if path.is_file():
                row = f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}'
                print(row)
                sha.write(row + '\n')
    print(f'raw_log={log_path}')
    print(f'hash_manifest={manifest}')
    return status


# 命令行入口：默认使用本脚本上两级 student_work 作为结果目录，可显式改路径。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline-check', action='store_true', help='CPU reference/algebra controls only; no CUDA')
    parser.add_argument('--work-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'runs')
    parser.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
