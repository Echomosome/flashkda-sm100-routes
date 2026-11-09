#!/usr/bin/env python3
"""Task 2 讨论点 1：指数范围探针及 CHUNK 资源的纸面预算。

目的：固定已激活 gate g=-alpha，找出 prefix 指数何时清零、倒数何时溢出。
默认：编译并运行独立 CUDA 标量指令实验；--paper-only：只算 CPU 数学公式。
本脚本不运行 FlashKDA 本体，不测性能，也没有实现 C32/C64 的完整 kernel。

建议先看 range_probe 的数值链，再看 paper_output 的预算；run_cuda 是运行包装。
读输出时同时看 zeros/nonfinite 和 last_k_restored，有限数也可能已有严重误差。
本文件补充中文注释；实测时的原始字节版本保留在 scripts/archive/ 中。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


# 【1. CUDA 探针源码】Python 将此字符串写入 .cu，再由 nvcc 编译。
CUDA_SOURCE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr) do { \
  cudaError_t err = (expr); \
  if (err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA_ERROR line=%d message=%s\n", __LINE__, cudaGetErrorString(err)); \
    return 2; \
  } \
} while (0)

// 每个 Job 是一个 chunk 长度和恒定 gate 强度；alpha 用自然对数单位。
struct Job { int chunk; float alpha; };
// first_* 从 1 开始，0 表示未观察到；last_* 是最后一个 token 的中间量。
struct Result {
  int decay_zero_count, inverse_nonfinite_count, restored_nonfinite_count;
  int first_decay_zero, first_inverse_nonfinite, first_restored_nonfinite;
  float prefix_log2, last_decay, last_inverse, last_k_inverse;
  float last_k_restored, last_rebased, k_component;
};

// 实际执行 PTX 近似 2^x 指令；FTZ 将该操作涉及的 FP32 次正规数清零。
// 使用这条 device 指令，才能核对真实范围行为，而非只算 CPU 数学 exp。
__device__ __forceinline__ float ex2_ftz(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// 一个线程负责一个 case；线程内部按 token 顺序累加，保留 FP32 舍入顺序。
// 输出只覆盖 exponent/BF16 子路径，不包括 MMA、求逆或 recurrent state。
__global__ void range_probe(const Job* jobs, Result* results, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const Job job = jobs[index];
  // g=-alpha 已经过激活；此处绕过 sigmoid，只考察指定 gate 的压力输入。
  // 乘 log2(e) 后可用 2^x 表示自然指数 exp(g)。
  const float gate_log2 = -job.alpha * 1.4426950408889634f;
  float total = 0.0f;
  for (int row = 0; row < job.chunk; ++row) total += gate_log2;
  // total 是整段 prefix；先求整段 decay，供每个 token 的恢复因子使用。
  // RN 为舍入到最近值；k 取 D=128 单位向量的一个分量，再转成 BF16。
  const __nv_bfloat16 total_decay = __float2bfloat16_rn(ex2_ftz(total));
  const __nv_bfloat16 k = __float2bfloat16_rn(1.0f / sqrtf(128.0f));
  Result result{};
  result.k_component = __bfloat162float(k);
  float prefix = 0.0f;
  for (int row = 0; row < job.chunk; ++row) {
    prefix += gate_log2;
    // 被测表示：分别计算很小的 decay 与很大的 inverse，再做 BF16 乘法。
    // k_restored=(k*2^(-prefix))*2^total；有限范围内也可能受中间舍入影响。
    const __nv_bfloat16 decay = __float2bfloat16_rn(ex2_ftz(prefix));
    const __nv_bfloat16 inverse = __float2bfloat16_rn(ex2_ftz(-prefix));
    const __nv_bfloat16 k_inverse = __hmul(k, inverse);
    const __nv_bfloat16 k_restored = __hmul(k_inverse, total_decay);
    // 代数对照先合并指数：k*2^(total-prefix)，避免生成单独的大倒数。
    // 最后一个 token 有 total=prefix，目标就是 k；这仅验证该因子，不是完整 rescale。
    const __nv_bfloat16 rebased = __hmul(k, __float2bfloat16_rn(ex2_ftz(total-prefix)));
    const float decay_f = __bfloat162float(decay);
    const float inverse_f = __bfloat162float(inverse);
    const float restored_f = __bfloat162float(k_restored);
    // 清零与 NaN/Inf 分开统计：错误变成 0 仍是有限值，只看 isfinite 会漏掉。
    if (decay_f == 0.0f) {
      ++result.decay_zero_count;
      if (result.first_decay_zero == 0) result.first_decay_zero = row+1;
    }
    if (!isfinite(inverse_f)) {
      ++result.inverse_nonfinite_count;
      if (result.first_inverse_nonfinite == 0) result.first_inverse_nonfinite = row+1;
    }
    if (!isfinite(restored_f)) {
      ++result.restored_nonfinite_count;
      if (result.first_restored_nonfinite == 0) result.first_restored_nonfinite = row+1;
    }
    result.prefix_log2 = prefix;
    result.last_decay = decay_f;
    result.last_inverse = inverse_f;
    result.last_k_inverse = __bfloat162float(k_inverse);
    result.last_k_restored = restored_f;
    result.last_rebased = __bfloat162float(rebased);
  }
  results[index] = result;
}

// host 入口：记录设备属性，构造 4×8 个 case，运行后逐行打印数值。
// 设备 shared-memory 上限仅供资源预算参考，不代表某个 kernel 已采用最大配置。
int main() {
  int device = 0, driver = 0, runtime = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  int smem_block_optin = 0, smem_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  std::printf("evidence=REAL_CUDA_INSTRUCTION_PROBE\n");
  std::printf("device=%s capability=%d.%d sm_count=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  std::printf("cuda_driver=%d cuda_runtime=%d\n", driver, runtime);
  std::printf("device_max_smem_block_optin=%d device_max_smem_per_sm=%d\n", smem_block_optin, smem_sm);
  std::printf("scope=constant_activated_gate; D=128; ex2.approx.ftz.f32; bf16_RN; no_MMA; no_latency\n");
  std::printf("first_prefix_indices_are_1_based; zero_means_not_observed\n");
  const int chunks[] = {16, 17, 18, 32, 34, 35, 36, 64};
  const float alphas[] = {0.0f, 1.0f, 2.5f, 5.0f};
  std::vector<Job> jobs;
  for (float alpha : alphas) for (int chunk : chunks) jobs.push_back({chunk, alpha});
  std::vector<Result> results(jobs.size());
  Job* device_jobs = nullptr;
  Result* device_results = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_jobs), jobs.size()*sizeof(Job)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_results), results.size()*sizeof(Result)));
  CUDA_CHECK(cudaMemcpy(device_jobs, jobs.data(), jobs.size()*sizeof(Job), cudaMemcpyHostToDevice));
  // 32 个 case 用一个 64-thread block；多出的线程由 count 边界检查返回。
  range_probe<<<1, 64>>>(device_jobs, device_results, static_cast<int>(jobs.size()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(results.data(), device_results, results.size()*sizeof(Result), cudaMemcpyDeviceToHost));
  std::puts("alpha,C,zeros_decay,nonfinite_inverse,nonfinite_restored,first_zero,first_bad_inverse,first_bad_restored,last_prefix_log2,last_decay,last_inverse,last_k_inverse,last_k_restored,last_rebased,k_component");
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const Result& r = results[i];
    std::printf("%.1f,%d,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
      jobs[i].alpha, jobs[i].chunk, r.decay_zero_count, r.inverse_nonfinite_count,
      r.restored_nonfinite_count, r.first_decay_zero, r.first_inverse_nonfinite,
      r.first_restored_nonfinite, r.prefix_log2, r.last_decay, r.last_inverse,
      r.last_k_inverse, r.last_k_restored, r.last_rebased, r.k_component);
  }
  CUDA_CHECK(cudaFree(device_jobs));
  CUDA_CHECK(cudaFree(device_results));
  std::puts("execution_completed=yes; nonfinite_stress_results_are_observations_not_test_failures");
  return 0;
}
'''


# 【2. 纸面预算】只在 CPU 算公式，不能充当编译后的资源大小或实机耗时。
def paper_output():
    """输出指数边界、求逆 FLOP、workspace 和 shared-memory 的假设预算。"""
    bf16_max = (2.0 - 2.0**-7) * 2.0**127
    normal_min = 2.0**-126
    bf16_subnormal_min = 2.0**-133
    print('evidence=OFFLINE_PAPER_ARITHMETIC; gpu_executed=no')
    print('assumptions=fixed_aligned,T8192,H96,D128,lower_bound_minus5; no_rescale; same_stage_counts')
    print(f'bf16_max={bf16_max:.12e} ln_max={math.log(bf16_max):.9f}')
    print(f'fp32_bf16_min_normal={normal_min:.12e} minus_ln_min_normal={-math.log(normal_min):.9f}')
    print(f'bf16_min_subnormal={bf16_subnormal_min:.12e}; ex2_FTZ_uses_fp32_normal_boundary')
    print('C,worst_abs_prefix,exp_minus5C,exp_plus5C,mean_alpha_at_FTZ_boundary,mean_alpha_at_bf16_max,neumann_dense_gemms,neumann_flop_per_chunk,neumann_fixed_total_ratio,k1_mma_flop_per_chunk,k2_mma_flop_per_chunk,workspace_MiB,k2_smem_budget_bytes')
    # 固定 T=8192、H=96、D=128；C 增大时 chunk 数按 T/C 减少。
    for c in (16, 32, 64):
        d, t, h = 128, 8192, 96
        # 每个倍增阶段：一次 Lpower 平方 + 一次 R*Lpower，共 2(log2 C-1) 个 GEMM。
        # dense C×C GEMM 计 2*C^3 FLOP；这里是运算量，不是时间预测。
        inverse_gemms = 2 * (c.bit_length() - 2)
        inverse_flop = inverse_gemms * 2 * c**3
        k1_flop = 4*c*c*d + inverse_flop
        k2_flop = 6*c*d*d + 4*c*c*d
        # 只算有效 tile 数据；不包含 allocator 余量、外部输入输出或 host 的保守上界。
        workspace = (6*c*d + 4*d + 4*c*c) * (t//c) * h
        input_stage = 8*c*d + 4*c*c + 640
        pipeline_data = 3*input_stage + 4*c*d
        # state + union 的较大分支 + 末尾对齐预算；union 的两分支不能相加。
        smem = 2*d*d + max(pipeline_data, 4*d*d) + 128
        fixed_inv_ratio = (inverse_flop*(t//c)) / (49152*(t//16))
        print(f'{c},{5*c},{math.exp(-5*c):.9e},{math.exp(5*c):.9e},'
              f'{-math.log(normal_min)/c:.9f},{math.log(bf16_max)/c:.9f},'
              f'{inverse_gemms},{inverse_flop},{fixed_inv_ratio:.9f},'
              f'{k1_flop},{k2_flop},{workspace/2**20:.3f},{smem}')
    print('alpha5_first_integer_prefix_below_fp32_normal=18 (analytical, not measured)')
    print('alpha5_first_integer_prefix_above_bf16_max=18 (analytical, not measured)')
    print('limits=RN_conversion_boundary_and_approx_error_require_device_probe; budgets_are_not_compiled_sizes')


# 【3. 运行包装】创建独立 UTC 目录，编译、运行并保存原始日志，算法不在此处。
def run_cuda(args):
    work = args.work_dir.resolve()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw_dir = work / 'logs/raw'
    build = work / 'build' / f'{run_id}_task2_chunk_range'
    raw_dir.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=False)
    log_path = raw_dir / f'{run_id}_task2_chunk_range.log'
    before = raw_dir / f'{run_id}_task2_chunk_range_gpu_before.csv'
    after = raw_dir / f'{run_id}_task2_chunk_range_gpu_after.csv'
    source = build / 'range_probe.cu'
    binary = build / 'range_probe'
    artifacts = [Path(__file__).resolve(), source, before, after, log_path]
    status = 1
    attempted_gpu = False
    with log_path.open('x') as log:
        # 终端和文件写入相同内容，失败记录也保留。
        def emit(value):
            print(value, flush=True)
            log.write(value + '\n')
            log.flush()

        # 参数列表直接传子进程，合并 stdout/stderr，并检查退出码。
        def command(argv, timeout):
            emit('command=' + shlex.join([str(part) for part in argv]))
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=timeout, check=False)
            if proc.stdout:
                emit(proc.stdout.rstrip('\n'))
            emit(f'command_exit={proc.returncode}')
            if proc.returncode:
                raise RuntimeError(f'command failed with exit {proc.returncode}')
            return proc.stdout

        # 前后 GPU 进程快照只反映两个采样时刻，不是全过程隔离证明。
        def snapshot(path):
            proc = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                   '--format=csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=20, check=False)
            with path.open('x') as f:
                f.write(proc.stdout)
            emit('gpu_snapshot=' + str(path))
            emit(proc.stdout.rstrip('\n'))
            if proc.returncode:
                raise RuntimeError(f'nvidia-smi failed with exit {proc.returncode}')

        emit('purpose=Task2_discussion1_exponent_range; not_FlashKDA_benchmark')
        emit(f'utc_run_id={run_id}')
        emit(f'python={sys.executable}')
        emit(f'script={Path(__file__).resolve()}')
        emit(f'slurm_job_id={os.environ.get("SLURM_JOB_ID", "missing")}')
        emit(f'raw_log={log_path}')
        try:
            # 默认模式必须在用户已有的 Slurm GPU 会话中运行，不自行分配资源。
            if not os.environ.get('SLURM_JOB_ID'):
                raise RuntimeError('Run inside the existing Slurm GPU allocation.')
            if not args.nvcc.is_file() or not os.access(args.nvcc, os.X_OK):
                raise RuntimeError(f'nvcc not executable: {args.nvcc}')
            if shutil.which('nvidia-smi') is None:
                raise RuntimeError('nvidia-smi unavailable')
            with source.open('x') as f:
                f.write(CUDA_SOURCE)
            command([str(args.nvcc), '--version'], timeout=20)
            command([str(args.nvcc), '-std=c++17', '-O2', '-arch=sm_103a',
                     '--fmad=false', str(source), '-o', str(binary)], timeout=240)
            artifacts.insert(2, binary)
            snapshot(before)
            attempted_gpu = True
            command([str(binary)], timeout=60)
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
            # exit=0 只表示程序完成；压力输入产生的 NaN/Inf 仍需按数值结果解释。
            emit(f'probe_exit={status}')
            emit('scope=CUDA_exponent_and_bf16_operations_only; FlashKDA_C32_C64_not_implemented')
    # 关闭日志后计算最终哈希；注释改变也会改变脚本/.cu 哈希，旧版本另行归档。
    sha_path = log_path.with_name(log_path.name + '.sha256')
    with sha_path.open('x') as sha:
        for path in artifacts:
            if path.is_file():
                row = f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}'
                print(row)
                sha.write(row + '\n')
    print(f'raw_log={log_path}')
    print(f'hash_manifest={sha_path}')
    return status


# 命令行入口：--paper-only 为纯 CPU；默认分支执行真实 CUDA 探针。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-only', action='store_true', help='CPU scalar formulas only; no CUDA or file writes')
    parser.add_argument('--work-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'runs')
    parser.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    args = parser.parse_args()
    if args.paper_only:
        paper_output()
        return 0
    return run_cuda(args)


if __name__ == '__main__':
    raise SystemExit(main())
