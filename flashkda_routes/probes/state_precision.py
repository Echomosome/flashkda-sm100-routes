#!/usr/bin/env python3
"""Task2 讨论点5：将状态存储误差与完整 FlashKDA 误差分开测量。

阅读顺序：make_fixture → reference_step/run_references → run_cuda → main。
--mode offline：只在 CPU/NumPy 上运行参考、舍入对照和自检，不加载 CUDA。
--mode cuda：在相同输入上运行已安装的 FlashKDA；不改原 kernel，不做计时。
默认 B=1、T=8192、H=4、D=128；四个 head 是四种压力条件，并非 H=96 基准。

R64：FP64 逐 token 数学递推；R32：同一预处理输入转 FP32 后递推。
Q16：FP64 逐 token 递推，仅每16 token/最终尾块将 state 经 FP32 RN→BF16 RN。
Q16 是状态存储的隔离模型，不模拟原 kernel 的 MMA、近似指数和分块求逆。
GPU 的 fp32-state 参数仅核对 I/O 路径；源码内 state 仍为 BF16。

每次新建 UTC 日志/结果目录；保存源文件身份、全部指标、GPU数组与哈希。
有限的大误差是观测结果，不擅自定模型验收阈值；结构/参考自检失败才中止。
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import traceback

import numpy as np


CHUNK = 16
D = 128
H = 4
EPS = 1e-6
HEADS = ("random_decay", "random_weak", "coherent_near_identity", "random_slow_write")


def emit(tag, **values):
    """每条记录独立 JSON，保留字段类型，避免 CSV 中逗号/数组难读。"""
    print(tag + " " + json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bf16_round(value):
    """软件 RN-even：先按 FP32 存储，再把低16位舍入掉，返回 FP32 容器。

    本函数不做 FTZ；它只表示存储舍入，不能代表 ex2.approx.ftz 指令。
    NaN 单独保留为 quiet NaN，防止位级加法把大 NaN 回绕为有限数。
    """
    x = np.asarray(value, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    rounded = np.where(np.isnan(x), bits | np.uint32(0x00400000), rounded).astype(np.uint32)
    return rounded.view(np.float32)


def bf16_bits(value):
    return (bf16_round(value).view(np.uint32) >> 16).astype(np.uint16)


def make_fixture(length, seed, initial):
    """所有分支使用同一份已量化输入；各 head 的变量不混在总体平均里。

    head2 重复同一个 k，q 读取未被 k 更新的 state 行，以暴露微小衰减丢失。
    其余 head 使用带正负输入；beta 较小的 head3 保留较长状态记忆。
    生成顺序固定，但不同 length 不是同一个随机流的可比前缀；一次运行内
    所有 checkpoint 都切取这份完整数组的前缀，避免换输入比较。
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    shape = (length, H, D)
    raw = {name: bf16_round(rng.standard_normal(shape) * scale)
           for name, scale in (("q", 1.0), ("k", 1.0), ("v", 0.125))}
    raw["k"][:, 2, :] = 0
    raw["k"][:, 2, 0] = 1
    raw["q"][:, 2, :] = 0
    raw["q"][:, 2, 1] = 1
    # gate 原始 logit；A_log=0、dt_bias=0、lower_bound=-5。
    raw["g"] = np.broadcast_to(np.array([-2, -8, -16, -16], np.float32)[None, :, None], shape).copy()
    raw["beta"] = np.broadcast_to(np.array([0, 0, 0, -4], np.float32), (length, H)).copy()
    state = bf16_round(rng.standard_normal((H, D, D)) * 0.125)
    if initial == "zero":
        state.fill(0)
    else:
        # q=e1 读到的整行精确为1/8，便于另用闭式解核对遗忘。
        state[2, 1, :] = 0.125
    raw["initial_state_KV"] = state
    return raw


def activate(raw):
    """数学预处理全部在 FP64：同一输入，规范化含与源码相同的 eps。

    此处刻意不复刻 GPU 的 BF16 规范化写回/近似 sigmoid；完整 GPU 差值
    将包含这些算法实现差异，Q16-R64 则仅改变状态存储舍入。
    """
    q, k, v = (raw[name].astype(np.float64) for name in ("q", "k", "v"))
    q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + EPS)
    k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + EPS)
    log_g = -5.0 / (1.0 + np.exp(-raw["g"].astype(np.float64)))
    beta = 1.0 / (1.0 + np.exp(-raw["beta"].astype(np.float64)))
    return q, k, v, log_g, beta


def reference_step(state, q, k, v, decay, beta):
    """状态用逻辑 [H,K,V]：先遗忘，再 delta update，最后计算当前输出。

    对应固定 FLA naive.py 的递推方程；用广播/逐轴求和避免 BLAS 线程调度。
    不原地修改传入 state，方便独立检验每一步的误差传播。
    """
    faded = state * decay[..., None]
    residual = v - np.sum(k[..., None] * faded, axis=-2)
    updated = faded + (beta[..., None] * k)[..., None] * residual[..., None, :]
    out = np.sum(q[..., None] * updated, axis=-2) * np.asarray(D ** -0.5, dtype=state.dtype)
    return updated, out


def checkpoints(length):
    return sorted({n for n in (16, 128, 1024, 4096, 8192, length) if n <= length})


def self_checks():
    """有独立答案的必要自检：RN偶数规则、矩阵方向、舍入时刻与遗忘闭式解。"""
    ties = np.array([1, 1 + 2**-8, 1 + 3 * 2**-8, -1 - 2**-8], np.float32)
    np.testing.assert_array_equal(bf16_round(ties), [1, 1, 1 + 2**-6, -1])
    np.testing.assert_array_equal(bf16_round(bf16_round(ties)), bf16_round(ties))
    assert np.isnan(bf16_round(np.array([np.nan], np.float32)))[0]
    # 非方阵、非对称初值：K=3,V=2，防止 D=128 方阵掩盖状态转置错误。
    s = np.arange(6, dtype=np.float64).reshape(1, 3, 2) / 8
    q = np.array([[0.25, -0.5, 0.75]])
    k = np.array([[0.25, 0.5, -0.25]])
    decay = np.array([[0.9, 0.8, 0.7]])
    v, beta = np.array([[0.125, -0.25]]), np.array([0.5])
    actual, out = reference_step(s, q, k, v, decay, beta)
    matrix = (np.eye(3) - beta[0] * np.outer(k[0], k[0])) @ np.diag(decay[0])
    expected = matrix @ s[0] + beta[0] * np.outer(k[0], v[0])
    np.testing.assert_allclose(actual[0], expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(out[0], q[0] @ expected * D**-0.5, rtol=1e-14, atol=1e-15)
    # 强制采用与主循环相同的 Q16 次序：最后一个 token 的输出先于 state 舍入。
    s = np.full((1, 1, 1), 0.125, np.float64)
    decay = np.array([[np.exp(-1e-5)]])
    for index in range(CHUNK):
        s, out = reference_step(s, np.ones((1, 1)), np.zeros((1, 1)),
                                np.zeros((1, 1)), decay, np.zeros(1))
    before = s.copy()
    s = bf16_round(s).astype(np.float64)
    np.testing.assert_allclose(before.item(), 0.125 * np.exp(-CHUNK * 1e-5), rtol=1e-14)
    assert s.item() == 0.125 and out.item() < s.item() * D**-0.5
    emit("SELF_CHECK", rounding="PASS", non_square_state_orientation="PASS",
         analytic_decay="PASS", output_before_boundary_rounding="PASS")


def metric_rows(tag, actual, reference, token, records):
    """逐 head 保存绝对/RMS/相对误差；不使用近零元素的逐点相对误差。

    输入 [H,...]。NRMSE 分母固定为 ref_rms+1e-12，并同时报告 ref_rms。
    非有限值不会被 nanmean 忽略；该行标明数量，其余指标为空。
    """
    for h, name in enumerate(HEADS):
        a, r = np.asarray(actual[h], np.float64), np.asarray(reference[h], np.float64)
        row = dict(comparison=tag, token=int(token), head=h, fixture=name,
                   elements=int(a.size), actual_nonfinite=int(np.count_nonzero(~np.isfinite(a))),
                   reference_nonfinite=int(np.count_nonzero(~np.isfinite(r))))
        if row["actual_nonfinite"] or row["reference_nonfinite"]:
            row.update(max_abs=None, rms_error=None, ref_rms=None, nrmse=None, p99_abs=None)
        else:
            error = np.abs(a - r)
            rms = float(np.sqrt(np.mean(error * error)))
            ref_rms = float(np.sqrt(np.mean(r * r)))
            row.update(max_abs=float(error.max()), rms_error=rms, ref_rms=ref_rms,
                       nrmse=rms / (ref_rms + 1e-12), p99_abs=float(np.quantile(error, 0.99)))
        records.append(row)
        emit("ERROR_METRIC", **row)


def run_references(raw, records):
    """一次逐 token 计算三个对照；完整输出保留在内存，state 只存检查点。

    误差上界来自 ||(I-beta kk^T)diag(exp(g))||2 <= max(exp(g))。
    前提为 beta∈[0,1]、||k||2<=1、g<=0；这只约束该数学递推。
    bound 是实测舍入扰动范数的累计上界，不是模型精度保证。
    """
    q, k, v, log_g, beta = activate(raw)
    length = q.shape[0]
    decay = np.exp(log_g)
    assert np.max(np.sum(k * k, axis=-1)) <= 1 + 1e-14
    assert np.all((beta >= 0) & (beta <= 1)) and np.all(log_g <= 0)
    q32, k32, v32, d32, b32 = (x.astype(np.float32) for x in (q, k, v, decay, beta))
    states = {"R64": raw["initial_state_KV"].astype(np.float64),
              "R32": raw["initial_state_KV"].copy(),
              "Q16": raw["initial_state_KV"].astype(np.float64)}
    outputs = {name: np.empty((length, H, D), dtype=s.dtype) for name, s in states.items()}
    saved, bound = {}, np.zeros(H)
    checkpoint_set = set(checkpoints(length))
    for i in range(length):
        states["R64"], outputs["R64"][i] = reference_step(states["R64"], q[i], k[i], v[i], decay[i], beta[i])
        states["R32"], outputs["R32"][i] = reference_step(states["R32"], q32[i], k32[i], v32[i], d32[i], b32[i])
        states["Q16"], outputs["Q16"][i] = reference_step(states["Q16"], q[i], k[i], v[i], decay[i], beta[i])
        bound *= np.max(decay[i], axis=-1)
        # tail 只在整条序列结束时写回；内部 checkpoint 均对齐16，避免额外舍入。
        if (i + 1) % CHUNK == 0 or i + 1 == length:
            rounded = bf16_round(states["Q16"]).astype(np.float64)
            bound += np.sqrt(np.sum((rounded - states["Q16"])**2, axis=(-2, -1)))
            states["Q16"] = rounded
        if i + 1 in checkpoint_set:
            token = i + 1
            saved[token] = {name: value.copy() for name, value in states.items()}
            for name in ("R32", "Q16"):
                metric_rows(name + "_state_vs_R64", states[name], states["R64"], token, records)
            error = np.sqrt(np.sum((states["Q16"] - states["R64"])**2, axis=(-2, -1)))
            assert np.all(error <= bound + 1e-10), "state rounding propagation bound failed"
            emit("PROPAGATION_BOUND", token=token, measured_fro=error.tolist(), upper_bound=bound.tolist())
            if np.any(raw["initial_state_KV"][2, 1]):
                expected = raw["initial_state_KV"][2, 1].astype(np.float64) * np.exp(log_g[:token, 2, 1].sum())
                np.testing.assert_allclose(states["R64"][2, 1], expected, rtol=2e-11, atol=1e-13)
    # 每个窗口单独度量，避免全序列平均掩盖后段漂移。
    for end in checkpoints(length):
        start = max(0, end - 128)
        for name in ("R32", "Q16"):
            metric_rows(name + "_last128_output_vs_R64", outputs[name][start:end].swapaxes(0, 1),
                        outputs["R64"][start:end].swapaxes(0, 1), end, records)
    emit("REFERENCE_COMPLETE", mode="CPU_NUMPY", internal_dtypes={k: str(v.dtype) for k, v in states.items()},
         analytic_head2="PASS_if_nonzero_initial", precision="FP64_is_numerical_reference_not_exact_real")
    return outputs, saved


def snapshot(path):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    with path.open("x") as f:
        f.write(result.stdout)
    emit("GPU_SNAPSHOT", file=str(path), command_exit=result.returncode)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError("GPU process snapshot command failed")


def check_fla_naive(raw, c1_root):
    """用原 FLA naive 的短前缀交叉核对方程和布局；明确它内部是 FP32。

    在 CPU Torch 上执行，避免把参考检查冒称为 GPU benchmark。输入用 FP64，
    输出也可能是 FP64容器，但源码中间的 .to(torch.float) 仍然只有 FP32。
    """
    import torch
    torch.set_num_threads(1)
    naive = runpy.run_path(str(c1_root / "fla_kda_ref/naive.py"))["naive_recurrent_kda"]
    short = {key: value[:min(64, len(raw["q"]))] if key != "initial_state_KV" else value
             for key, value in raw.items()}
    q, k, v, g, beta = activate(short)
    arguments = [torch.from_numpy(x.copy()).unsqueeze(0) for x in (q, k, v, g, beta)]
    initial = torch.from_numpy(raw["initial_state_KV"].astype(np.float64)).unsqueeze(0)
    with torch.inference_mode():
        actual_o, actual_s = naive(*arguments, scale=D**-0.5, initial_state=initial, output_final_state=True)
    state = raw["initial_state_KV"].copy()
    expected_o = []
    # 原 naive 内部是先把 g 转 FP32 再求 exp，按其顺序做这个交叉核对。
    for i in range(len(q)):
        state, out = reference_step(state, q[i].astype(np.float32), k[i].astype(np.float32),
                                    v[i].astype(np.float32), np.exp(g[i].astype(np.float32)), beta[i].astype(np.float32))
        expected_o.append(out)
    a_o, a_s = actual_o[0].numpy(), actual_s[0].numpy()
    np.testing.assert_allclose(a_o, np.asarray(expected_o), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(a_s, state, rtol=2e-5, atol=2e-6)
    emit("FLA_NAIVE_CROSSCHECK", device="CPU_TORCH", tokens=len(q), status="PASS",
         source_internal_precision="FP32", max_abs_output=float(np.max(np.abs(a_o - expected_o))),
         max_abs_state=float(np.max(np.abs(a_s - state))))


def run_cuda(raw, outputs, saved, records, result_dir, c1_root):
    """运行原 FlashKDA 本体，用每个前缀独立重放取得真实 state 检查点。

    每次从同一初态开始，不把中途 API 额外分段混入原 fused recurrence。
    输出缓冲区预填 NaN，使漏写更容易被发现；FP32 I/O 对照使用同一BF16初值。
    """
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA mode requires an existing Slurm allocation")
    import torch
    import flash_kda
    import flash_kda_C
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    emit("DEVICE", name=prop.name, capability=[prop.major, prop.minor], SMs=prop.multi_processor_count,
         torch=torch.__version__, cuda=torch.version.cuda, extension=str(flash_kda_C.__file__),
         extension_sha256=sha256(flash_kda_C.__file__))
    check_fla_naive(raw, c1_root)
    with torch.inference_mode():
        gpu = {name: torch.from_numpy(raw[name]).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
               for name in ("q", "k", "v", "g", "beta")}
        # API 使用 [B,H,V,K]，数学参考使用 [H,K,V]；显式转置，不能只 reshape。
        s0 = torch.from_numpy(raw["initial_state_KV"].swapaxes(-2, -1).copy()).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
        a_log = torch.zeros(H, dtype=torch.float32, device="cuda")
        dt_bias = torch.zeros(H, D, dtype=torch.float32, device="cuda")
        result_arrays = {}
        length = len(raw["q"])
        full_output = full_state = None
        for token in [length] + [n for n in checkpoints(length) if n != length]:
            tensors = [gpu[name][:, :token].contiguous() for name in ("q", "k", "v", "g", "beta")]
            out = torch.full((1, token, H, D), float("nan"), dtype=torch.bfloat16, device="cuda")
            final = torch.full_like(s0, float("nan"))
            flash_kda.fwd(*tensors, D**-0.5, out, a_log, dt_bias, -5.0, initial_state=s0, final_state=final)
            torch.cuda.synchronize()
            actual_o = out[0].float().cpu().numpy()
            actual_s = final[0].float().cpu().numpy().swapaxes(-2, -1).copy()
            if token == length:
                full_output, full_state = actual_o.copy(), actual_s.copy()
                result_arrays["flash_output_bf16_bits"] = bf16_bits(actual_o)
            else:
                emit("PREFIX_REPLAY", token=token, differing_elements=int(np.count_nonzero(actual_o != full_output[:token])),
                     scope="consistency_observation_not_accuracy_threshold")
            for target in ("R64", "Q16"):
                metric_rows("FlashKDA_state_vs_" + target, actual_s, saved[token][target], token, records)
            start = max(0, token - 128)
            metric_rows("FlashKDA_last128_output_vs_R64", actual_o[start:].swapaxes(0, 1),
                        outputs["R64"][start:token].swapaxes(0, 1), token, records)
            result_arrays[f"flash_state_{token}_KV_bf16_bits"] = bf16_bits(actual_s)
        # 这里变的是 API 存储类型；不能把结果称为内部 FP32 recurrence。
        final32 = torch.full(s0.shape, float("nan"), dtype=torch.float32, device="cuda")
        out32 = torch.full_like(gpu["v"], float("nan"))
        flash_kda.fwd(*(gpu[name] for name in ("q", "k", "v", "g", "beta")),
                      D**-0.5, out32, a_log, dt_bias, -5.0, initial_state=s0.float(), final_state=final32)
        torch.cuda.synchronize()
        state32 = final32[0].cpu().numpy().swapaxes(-2, -1).copy()
        result_arrays["flash_fp32_api_final_KV"] = state32
        emit("STATE_IO_CONTROL", output_differences=int(np.count_nonzero(out32[0].float().cpu().numpy() != full_output)),
             state_differences=int(np.count_nonzero(state32 != full_state)),
             fp32_api_values_outside_bf16_grid=int(np.count_nonzero(state32 != bf16_round(state32))),
             meaning="FP32_API_does_not_change_internal_BF16_state")
        metric_rows("FlashKDA_all_output_vs_R64", full_output.swapaxes(0, 1), outputs["R64"].swapaxes(0, 1), length, records)
        # 实机结果和原始输入均保存；不只留下汇总，以便以后重新计算指标。
        result_arrays.update({"input_" + name + "_bf16_bits": bf16_bits(value) for name, value in raw.items()})
        result_arrays.update({f"reference_{name}_{token}_KV": value
                              for token, data in saved.items() for name, value in data.items()})
        with (result_dir / "gpu_inputs_outputs_states.npz").open("xb") as f:
            np.savez_compressed(f, **result_arrays)
        emit("CUDA_COMPLETED", status="EXECUTED", performance="NOT_MEASURED",
             accuracy_acceptance="NOT_DEFINED_BY_THIS_PROBE")


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("offline", "cuda"), default="offline")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--initial", choices=("nonzero", "zero"), default="nonzero")
    parser.add_argument("--work-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "runs",
                        help="output root; its parent must contain FlashKDA/ and fla_kda_ref/")
    args = parser.parse_args()
    if not 16 <= args.length <= 131072:
        parser.error("length must be in [16,131072]")
    script = Path(__file__).resolve()
    student = args.work_dir.resolve()
    c1_root = student.parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label = run_id + "_task2_state_precision_" + args.mode
    result_dir = student / "results" / label
    result_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = student / "logs/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / (label + ".log")
    before, after = raw_dir / (label + "_gpu_before.csv"), raw_dir / (label + "_gpu_after.csv")
    records, exit_code = [], 0
    # 整段异常和 stderr 一并保留；目录/文件独占创建，不覆盖任何旧结果。
    with log_path.open("x") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        emit("RUN", purpose="Task2_discussion5_state_precision", mode=args.mode, utc_run_id=run_id,
             python=sys.executable, host=platform.node(), numpy=np.__version__,
             slurm_job_id=os.environ.get("SLURM_JOB_ID"), script=str(script), raw_log=str(log_path))
        emit("CONFIG", B=1, T=args.length, H=H, D=D, CHUNK=CHUNK, seed=args.seed, initial=args.initial,
             checkpoints=checkpoints(args.length), fixture_heads=list(HEADS), no_benchmark=True)
        try:
            for rel in ("FlashKDA/csrc/smxx/fwd_kernel2.cuh", "FlashKDA/tests/torch_ref.py",
                        "fla_kda_ref/naive.py", "fla_kda_ref/fused_recurrent.py"):
                path = c1_root / rel
                emit("SOURCE", path=str(path), sha256=sha256(path))
            self_checks()
            raw = make_fixture(args.length, args.seed, args.initial)
            for name, value in raw.items():
                emit("INPUT", name=name, shape=list(value.shape), dtype=str(value.dtype),
                     sha256_contiguous_bytes=hashlib.sha256(value.tobytes()).hexdigest())
            outputs, saved = run_references(raw, records)
            if args.mode == "cuda":
                snapshot(before)
                try:
                    run_cuda(raw, outputs, saved, records, result_dir, c1_root)
                finally:
                    snapshot(after)
            else:
                emit("OFFLINE_ONLY", CUDA_executed=False, FlashKDA_executed=False,
                     scope="CPU_reference_and_boundary_storage_model")
            nonfinite_rows = sum(bool(r["actual_nonfinite"] or r["reference_nonfinite"]) for r in records)
            emit("COMPLETED", records=len(records), nonfinite_rows=nonfinite_rows,
                 execution_status="SUCCESS", accuracy_acceptance="NOT_INFERRED_FROM_EXIT_CODE")
        except Exception:
            exit_code = 1
            traceback.print_exc()
        finally:
            with (result_dir / "metrics.json").open("x") as f:
                json.dump(dict(config=vars(args), records=records, exit_code=exit_code), f,
                          ensure_ascii=False, indent=2, allow_nan=False, default=str)
            emit("EXIT", probe_exit=exit_code, result_dir=str(result_dir))
    files = [script, log_path, *sorted(result_dir.iterdir())]
    files.extend(p for p in (before, after) if p.exists())
    manifest = log_path.with_suffix(log_path.suffix + ".sha256")
    with manifest.open("x") as f:
        for path in files:
            line = sha256(path) + "  " + str(path)
            print(line)
            f.write(line + "\n")
    print("raw_log=" + str(log_path))
    print("hash_manifest=" + str(manifest))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
