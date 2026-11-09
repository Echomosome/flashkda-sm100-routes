#!/usr/bin/env python3
"""Task 3：完整 FlashKDA 的独立 value 列分片候选，第一轮只编译与验正确性。

阅读顺序：generate_candidate（源码差异）→ offline_audit（离线能证明什么）
→ cuda_worker（B300 编译/布局/对拍）→ main（日志与证据归档）。

--mode offline 不导入 torch、不编译 CUDA、不使用 GPU；会实际生成候选源码和 patch。
--mode cuda 要求已有 Slurm GPU allocation、题目 .venv、原 CUTLASS 子模块。
仅向 student_work/build 写新目录；不安装扩展、不修改 FlashKDA 基线或已测脚本。
本轮没有 timer：通过后才能讨论完整 KDA 的速度，不能预填收益。

PATH 修订：依据20260910T151620047153818Z实机诊断，仅为构建子进程补入
当前Python的sys.prefix/bin，并记录实际Ninja路径/版本；不改父shell或安装包。
CUDA生成器、编译参数和正确性判据保持原版；旧task3_colsplit_probe.py保留。
"""

import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import traceback


# 固定题目快照；不把未知版本上的近似文本匹配偷偷当成同一实现。
SOURCE_SHA256 = {
    "fwd.h": "462afefb5c1bfe89428166cd81fc70954606049a41a4b67376025a2b2f91d7cc",
    "flash_kda.cpp": "f736ab87c7c2e47fe416f8d04d62502a257a6ff32a0b23452c07e1bc65c87c04",
    "smxx/fwd_kernel1.cuh": "e0f369b51753d5e1a62b8435f0b692fdc856d87e52eb27d012c22b6575a0370f",
    "smxx/fwd_kernel2.cuh": "f8a634154ca26a18855adbd3db17d6208a874b16f173d0eb9814bfefff03b9b9",
    "smxx/fwd_launch.cu": "dbaf9ba867fc8c799cf147b7e17bd2aba4a08b1b63c5f7df0cca1173a0df5562",
    "smxx/utils.cuh": "211cc28d910758e2658a53f5bcda779f1076df281ff75c09ce9bd1638a54151a",
}
FLA_NAIVE_SHA256 = "60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016"


def emit(tag, **fields):
    """一行一个结构化记录；JSON 保留路径空格，不依赖 shell 拼接。"""
    print(tag + " " + json.dumps(fields, ensure_ascii=False, sort_keys=True), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def replace_exact(text, old, new, count=1):
    """每处变更检查命中数；源码不同即停下，避免漏改或误改 key 维。"""
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"patch anchor count {actual} != {count}: {old[:140]!r}")
    return text.replace(old, new)


LAYOUT_CHECK_CUDA = r'''
// Task3：在 host 上枚举真实 CuTe 布局，检查范围、唯一性、转置别名及 TMA 前置维。
// 这段检查须由 nvcc 编译后才能成立；Python 离线预算不是它的替代品。
template <class L, class LT, class TMA, int Rows, int Cols>
void c1_check_bf16_layout() {
    static_assert(cute::cosize_v<L> == Rows * Cols, "unexpected layout padding");
    std::vector<int> seen(cute::cosize_v<L>, 0);
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            int off = L{}(cute::make_coord(r, c));
            if (off < 0 || off >= int(seen.size()) || seen[off]++)
                throw std::runtime_error("CuTe layout out of range or aliases");
            if (off != LT{}(cute::make_coord(c, r)) ||
                off != TMA{}(cute::make_coord(0, r, c)))
                throw std::runtime_error("CuTe transpose/TMA layout mismatch");
        }
    }
}

std::vector<int64_t> c1_colsplit_layout_info() {
    using L = K2Layouts<128, 16>;
    c1_check_bf16_layout<L::StateSmemLayout, L::TransposedStateSmemLayout,
                        L::TMAStateSmemLayout, 64, 128>();
    c1_check_bf16_layout<L::VOLayout, L::TransposedVOLayout,
                        L::TMAVOLayout, 16, 64>();
    c1_check_bf16_layout<L::MMALayout, L::TransposedMMALayout,
                        L::TMAMMALayout, 16, 128>();
    using F = L::FP32StateSmemLayout;
    using FT = L::TMAFP32StateSmemLayout;
    static_assert(cute::cosize_v<F> == 64 * 128, "FP32 buffer size mismatch");
    std::vector<int> seen(cute::cosize_v<F>, 0);
    for (int r = 0; r < 64; ++r) {
        for (int c = 0; c < 128; ++c) {
            int off = F{}(cute::make_coord(r, c));
            if (off < 0 || off >= int(seen.size()) || seen[off]++ ||
                off != FT{}(cute::make_coord(0, r, c)))
                throw std::runtime_error("FP32 CuTe layout mismatch");
        }
    }
    return {int64_t(sizeof(SharedStorageK2<L, 3, 2>)), 128, 64, 128, 16, 2};
}
'''


def generate_candidate(c1_root, destination):
    """复制并具体改写完整 extension：K1 不变，K2 的 V/O/state 缩到半宽。

    key 维、CHUNK、所有 Phase 内浮点表达式保持原样；只改变列归属/布局/线程。
    返回原文与新文，供独立检查和可审阅 diff 使用。
    """
    source = c1_root / "FlashKDA/csrc"
    originals = {}
    for name, expected in SOURCE_SHA256.items():
        p = source / name
        if sha256(p) != expected:
            raise RuntimeError(f"baseline source hash mismatch: {p}")
        originals[name] = p.read_text()
        emit("BASELINE_SOURCE", file=str(p), sha256=expected)
    candidate = dict(originals)
    k2 = candidate["smxx/fwd_kernel2.cuh"]
    k2 = replace_exact(k2, '#include "utils.cuh"',
                       '#include "utils.cuh"\n#include "task3_state_cast.cuh"')
    k2 = replace_exact(k2, 'struct K2Layouts {', '''struct K2Layouts {
    // Task3：全局 K=V=128；本 CTA 仅保存 DV=64 个 value 列，key 仍为 D。
    static_assert(D == 128 && CHUNK == 16, "Task3 v1 only supports D128/C16");
    static constexpr int DV = D / 2;''')
    k2 = replace_exact(k2, '''    using VOLayout = MMALayout;
    using TransposedVOLayout = TransposedMMALayout;''', '''    // V/O 是 [CHUNK, DV]；KD/QD/KR 继续使用完整 [CHUNK, D]。
    using VOLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<DV>{}), LayoutLeft{}));
    using TransposedVOLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<DV>{}, Int<CHUNK>{}), LayoutRight{}));''')
    # 三个 state 视图分别为 [DV,D]、转置 [D,DV]、FP32 [DV,D]。
    for layout_name, shape in [
        ("StateSmemLayout", "Int<DV>{}, Int<D>{}"),
        ("TransposedStateSmemLayout", "Int<D>{}, Int<DV>{}"),
        ("FP32StateSmemLayout", "Int<DV>{}, Int<D>{}"),
    ]:
        begin = k2.index(f"    using {layout_name} = decltype(")
        end = k2.index("    ));", begin) + len("    ));")
        fragment = k2[begin:end]
        revised = replace_exact(fragment, "Int<D>{}, Int<D>{}", shape)
        k2 = k2[:begin] + revised + k2[end:]
    k2 = replace_exact(k2, "    using TMAStateSmemLayout = decltype(composition(", '''    // 单独保留完整 workspace tile 的 TMA 视图，不能沿用缩小后的 VO。
    using TMAMMALayout = decltype(composition(
        MMALayout{}.layout_a(), MMALayout{}.offset(), prepend(MMALayout{}.layout_b())
    ));
    using TMAStateSmemLayout = decltype(composition(''')
    k2 = replace_exact(k2, "    using TMAVOLayout = typename Layouts::TMAVOLayout;",
                      "    using TMAVOLayout = typename Layouts::TMAVOLayout;\n"
                      "    using TMAMMALayout = typename Layouts::TMAMMALayout;\n"
                      "    constexpr int DV = Layouts::DV;")
    k2 = replace_exact(k2, "constexpr int kComputeThreads = 128;", '''constexpr int kComputeThreads = DV;
    // 每 warp 仍处理 32 个 value 列；2 compute + 1 load + 1 store = 128 threads。
    static_assert(NumThreads == kComputeThreads + 64, "warp roles mismatch");''')
    k2 = replace_exact(k2, "    int head_idx = blockIdx.y;",
                      "    int head_idx = blockIdx.y;\n"
                      "    int value_offset = int(blockIdx.z) * DV;  // 分片 0/1：列 0/64")
    k2 = replace_exact(k2, "(seq_idx * H + head_idx, 0, 0)",
                      "(seq_idx * H + head_idx, value_offset, 0)", 4)
    k2 = replace_exact(k2, "make_shape(Int<1>{}, Int<D>{}, Int<D>{})",
                      "make_shape(Int<1>{}, Int<DV>{}, Int<D>{})", 4)
    for tensor in ("v", "out"):
        k2 = replace_exact(k2, f"g_{tensor}.layout()(head_idx, int(bos) + t * CHUNK, 0)",
                          f"g_{tensor}.layout()(head_idx, int(bos) + t * CHUNK, value_offset)")
        k2 = replace_exact(k2,
            f"make_shape(Int<1>{{}}, Int<CHUNK>{{}}, Int<D>{{}}), stride(g_{tensor}.layout())",
            f"make_shape(Int<1>{{}}, Int<CHUNK>{{}}, Int<DV>{{}}), stride(g_{tensor}.layout())")
    for tensor in ("k_decayed", "q_decayed", "k_restored"):
        k2 = replace_exact(k2, f"{tensor}.begin()), TMAVOLayout{{}})",
                          f"{tensor}.begin()), TMAMMALayout{{}})")
    k2 = replace_exact(k2, "* H * D + head_idx * D;", "* H * D + head_idx * D + value_offset;")
    k2 = replace_exact(k2, "col < D;", "col < DV;")
    k2 = replace_exact(k2, "// Only one thread (lane_predicate) runs here, so loop over all D",
                      "// Task3：只有 store leader 写本分片 DV 列；尾块不得写到下一序列。")
    k2 = replace_exact(k2, "smem_cvt_fp32_to_bf16<FP32StateSmemLayout, StateSmemLayout, D, NumThreads>",
                      "c1_rect_fp32_to_bf16<FP32StateSmemLayout, StateSmemLayout, DV, D, NumThreads>")
    k2 = replace_exact(k2, "smem_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, D, NumThreads>",
                      "c1_rect_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, DV, D, NumThreads>")
    k2 = k2.replace("_flash_kda_fwd_recurrence", "_c1_colsplit_fwd_recurrence")
    candidate["smxx/fwd_kernel2.cuh"] = k2

    # 沿用原 8×8 转换原子，仅把遍历范围扩展成 Rows×Cols；不修改 utils.cuh。
    utils = originals["smxx/utils.cuh"]
    casts = utils[utils.index("template <class FP32Layout, class BF16Layout, int D, int NumThreads>"):]
    casts = casts.replace("int D, int NumThreads", "int Rows, int Cols, int NumThreads")
    casts = replace_exact(casts, "constexpr int kBlocksPerDim = D / kBlock;",
                         "static_assert(Rows % 8 == 0 && Cols % 8 == 0, \"8x8 cast tiles\");\n"
                         "    constexpr int kBlocksPerDim = Cols / kBlock;", 2)
    casts = replace_exact(casts, "constexpr int kTotalBlocks = kBlocksPerDim * kBlocksPerDim;",
                         "constexpr int kTotalBlocks = (Rows / kBlock) * kBlocksPerDim;", 2)
    casts = casts.replace("smem_cvt_fp32_to_bf16", "c1_rect_fp32_to_bf16")
    casts = casts.replace("smem_cvt_bf16_to_fp32", "c1_rect_bf16_to_fp32")
    candidate["smxx/task3_state_cast.cuh"] = (
        '#pragma once\n#include "utils.cuh"\n'
        '// Task3：state 物理布局是 [value,key]，本 CTA 的 Rows=64、Cols=128。\n'
        '// 保留原 BF16 转换运算和 8×8 原子；修正方阵 helper 的总 tile 数。\n' + casts)

    launch = candidate["smxx/fwd_launch.cu"]
    launch = replace_exact(launch, '#include "fwd.h"', '''#include "fwd.h"
#include <vector>
#include <stdexcept>
// 独立扩展中的 K1 算法与源码不变，仅改 kernel 符号，避免和基线 DSO 混用。
#define _flash_kda_fwd_prepare _c1_colsplit_fwd_prepare
#define _flash_kda_build_tile_prefix _c1_colsplit_build_tile_prefix''')
    launch = replace_exact(launch, "    // TMA layouts for Kernel 2",
                          "    // Task3：只有原始 V 和输出 O 使用半宽 TMA；workspace 仍完整。\n"
                          "    using K2TMAVOLayout = typename K2L::TMAVOLayout;\n"
                          "    // TMA layouts for Kernel 2")
    for tensor in ("v", "out"):
        launch = replace_exact(launch, f"m_{tensor}, TMAVOLayout{{}})", f"m_{tensor}, K2TMAVOLayout{{}})")
    launch = replace_exact(launch, "constexpr int kK2Threads = 32 * 2 + 128;",
                           "constexpr int kK2Threads = 32 * 2 + K2L::DV;")
    launch = replace_exact(launch, "dim3 grid_k2(N, H);",
                          "dim3 grid_k2(N, H, 2);  // 两份独立 value 列，无跨 CTA state 归约")
    launch = launch.replace("_flash_kda_fwd_recurrence", "_c1_colsplit_fwd_recurrence")
    launch += LAYOUT_CHECK_CUDA
    candidate["smxx/fwd_launch.cu"] = launch
    # C++ host 符号也隔离；Python API 名称保持 fwd/get_workspace_size。
    for name in ("fwd.h", "flash_kda.cpp", "smxx/fwd_launch.cu"):
        candidate[name] = re.sub(r"\blaunch_fwd\b", "c1_colsplit_launch_fwd", candidate[name])
    cpp = candidate["flash_kda.cpp"]
    cpp = replace_exact(cpp, "#include <torch/extension.h>",
                        "#include <torch/extension.h>\n#include <pybind11/stl.h>\n#include <vector>")
    cpp = replace_exact(cpp, "void fwd(", "void c1_colsplit_fwd(")
    cpp = replace_exact(cpp, 'm.def("fwd", &fwd,', 'm.def("fwd", &c1_colsplit_fwd,')
    cpp = replace_exact(cpp, "int64_t get_workspace_size(", "int64_t c1_colsplit_get_workspace_size(")
    cpp = replace_exact(cpp, "(&get_workspace_size)", "(&c1_colsplit_get_workspace_size)")
    cpp = replace_exact(cpp, "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", '''// host 检查真实 CuTe 布局，返回已编译结构的 sizeof；不运行 GPU。
std::vector<int64_t> c1_colsplit_layout_info();
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layout_info", &c1_colsplit_layout_info);''')
    candidate["flash_kda.cpp"] = cpp

    destination.mkdir(parents=True, exist_ok=False)
    patch = []
    for name, revised in candidate.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(revised)
        patch.extend(difflib.unified_diff(originals.get(name, "").splitlines(True),
                                        revised.splitlines(True),
                                        fromfile="baseline/csrc/" + name,
                                        tofile="candidate/csrc/" + name))
    (destination.parent / "candidate.patch").write_text("".join(patch))
    (destination.parent / "source_manifest.json").write_text(json.dumps({
        "baseline": SOURCE_SHA256,
        "candidate": {name: sha256(destination / name) for name in candidate},
        "scope": "full_KDA_parallelism_candidate; CUDA_not_compiled_by_generation",
    }, indent=2) + "\n")
    return originals, candidate


def offline_audit(originals, candidate):
    """只验证源码不变量、逻辑地址归属和数学可分性，不冒充 CUDA 正确性。"""
    import numpy as np

    for name in ("smxx/fwd_kernel1.cuh", "smxx/utils.cuh"):
        assert originals[name] == candidate[name]
    start = "            // Fused MMA:"
    end = "\n#ifndef TMA_DISABLE_ALL\n    if (warp_role == WarpRole::STORE && lane_predicate)"
    a, b = originals["smxx/fwd_kernel2.cuh"], candidate["smxx/fwd_kernel2.cuh"]
    assert a[a.index(start):a.index(end)] == b[b.index(start):b.index(end)]
    emit("OFFLINE_SOURCE_CHECK", status="PASS", K1_and_utils="byte_identical",
         fused_MMA_and_epilogue="byte_identical", key_dimension=128,
         value_columns_per_CTA=64, compute_warps=2, io_warps=2)

    # 两个 CTA 的全局 state 地址不交叠，且恰好覆盖一个 head；FP32 cast 也恰好一次。
    state_owners = [(part * 64 + row) * 128 + col
                    for part in range(2) for row in range(64) for col in range(128)]
    assert sorted(state_owners) == list(range(128 * 128))
    cast_owners = []
    for tid in range(128):
        warp, lane = divmod(tid, 32)
        for tile in range(warp, 8 * 16, 4):
            br, bc = (tile // 16) * 8, (tile % 16) * 8
            for e in (lane * 2, lane * 2 + 1):
                cast_owners.append((br + e // 8) * 128 + bc + e % 8)
    assert sorted(cast_owners) == list(range(64 * 128))
    for lengths, heads in [([37, 37], 4), ([17, 33, 65], 4), ([1, 16, 17], 3)]:
        written = []
        bos = 0
        for length in lengths:
            for head in range(heads):
                for part in range(2):
                    for t in range(0, length, 16):
                        for row in range(min(16, length - t)):
                            base = (bos + t + row) * heads * 128 + head * 128 + part * 64
                            written.extend(range(base, base + 64))
            bos += length
        assert sorted(written) == list(range(sum(lengths) * heads * 128))
    emit("OFFLINE_OWNERSHIP_CHECK", status="PASS", state_elements=16384,
         rectangular_cast_elements=8192, output_cases=3,
         scope="logical_global_addresses; not_CuTe_or_TMA_execution")

    # state[v,k] 的更新只依赖同一 v 的所有 k，因此按 v 分片无需 state 归约。
    rng = np.random.default_rng(20260910)
    q = rng.normal(size=(37, 2, 128)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    v = rng.normal(size=q.shape).astype(np.float32)
    decay = np.exp(-rng.uniform(0, 0.2, size=q.shape)).astype(np.float32)
    beta = rng.uniform(0, 1, size=(37, 2)).astype(np.float32)
    initial = rng.normal(size=(2, 128, 128)).astype(np.float32) / 8

    def model(lo, hi):
        state = initial[:, lo:hi].copy()
        out = []
        for t in range(37):
            state *= decay[t, :, None, :]
            residual = v[t, :, lo:hi] - np.sum(state * k[t, :, None, :], axis=-1)
            state += (beta[t, :, None] * residual)[..., None] * k[t, :, None, :]
            out.append(np.sum(state * q[t, :, None, :], axis=-1) * 128 ** -0.5)
            if (t + 1) % 16 == 0 or t == 36:
                bits = state.view(np.uint32)
                state = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)
        return np.asarray(out), state

    full_o, full_s = model(0, 128)
    left_o, left_s = model(0, 64)
    right_o, right_s = model(64, 128)
    assert np.array_equal(full_o, np.concatenate([left_o, right_o], axis=-1))
    assert np.array_equal(full_s, np.concatenate([left_s, right_s], axis=1))
    emit("OFFLINE_ALGEBRA_CHECK", status="PASS", tokens=37, heads=2,
         model="scalar_recurrence_FP32_and_chunk_boundary_BF16_RN",
         scope="column_separability_only; not_fused_FlashKDA_numerical_emulation")


def snapshot(path):
    """保存进程快照；只观察，不改 clocks、不结束别人的作业。"""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                        "--format=csv"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path.write_text(r.stdout)
    print(r.stdout, end="", flush=True)
    emit("GPU_SNAPSHOT", file=str(path), exit=r.returncode)
    if r.returncode:
        raise RuntimeError("nvidia-smi snapshot failed")


def cuda_worker(args):
    """独立进程编译候选并对拍；所有 native compiler stderr 由父进程一起归档。"""
    # 在真实工作进程中核对继承的 PATH，再进入 torch/CUDA 构建。
    # 上游只修子进程环境，不依赖用户额外激活 .venv 或手动 export。
    ninja_path = shutil.which("ninja")
    emit("NINJA_PREFLIGHT", executable=ninja_path, python_prefix=sys.prefix)
    if ninja_path is None:
        raise RuntimeError("ninja is not executable in the build worker PATH")
    ninja_version = subprocess.run(
        ["ninja", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=10)
    emit("NINJA_VERSION", exit=ninja_version.returncode, output=ninja_version.stdout.strip())
    if ninja_version.returncode:
        raise RuntimeError(f"ninja --version failed: {ninja_version.returncode}")

    import importlib.util
    import runpy
    import torch
    from torch.utils.cpp_extension import load

    work = args.work.resolve()
    c1_root = work.parent
    build = args.worker.resolve()
    source = build / "csrc"
    cutlass = c1_root / "FlashKDA/cutlass"
    if not (cutlass / "include/cute/tensor.hpp").is_file():
        raise RuntimeError(f"missing existing CUTLASS headers: {cutlass}; no automatic download")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("this first candidate build is scoped to the B300 sm_103a run")
    props = torch.cuda.get_device_properties(0)
    emit("DEVICE", name=props.name, capability=[props.major, props.minor],
         SMs=props.multi_processor_count, torch=torch.__version__, torch_cuda=torch.version.cuda,
         python=sys.executable, numpy=__import__("numpy").__version__)
    subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], check=True)
    spec = importlib.util.find_spec("flash_kda_C")
    if spec is None or not spec.origin:
        raise RuntimeError("installed baseline flash_kda_C not found; use the Python environment with the pinned FlashKDA installed")
    baseline_so = Path(spec.origin)
    emit("BASELINE_BINARY", file=str(baseline_so), sha256=sha256(baseline_so))
    import flash_kda_C as baseline

    jit = build / "jit"
    jit.mkdir()
    # 与题目 setup.py 保持相同优化/半精度宏，显式保留 a 架构条件特性。
    # C++ 标准由当前 PyTorch 的 extension 工具补齐；不强制降到 C++17。
    nvcc_flags = ["-O3", "-gencode=arch=compute_103a,code=sm_103a",
                  "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                  "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                  "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
                  "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills",
                  "-lineinfo", "--threads", "2", "-Xcompiler=-fvisibility=hidden"]
    emit("BUILD", mode="separate_torch_JIT_extension; no_install", nvcc_flags=nvcc_flags,
         max_jobs=os.environ["MAX_JOBS"], cutlass=str(cutlass))
    candidate = load(
        name="c1_colsplit_" + re.sub(r"[^0-9]", "", build.name),
        sources=[str(source / "flash_kda.cpp"), str(source / "smxx/fwd_launch.cu")],
        extra_include_paths=[str(source), str(source / "smxx"), str(cutlass / "include"),
                             str(cutlass / "examples/common"), str(cutlass / "tools/util/include")],
        extra_cflags=["-O3", "-Wno-psabi", "-fvisibility=hidden"],
        extra_cuda_cflags=nvcc_flags, build_directory=str(jit), verbose=True)
    info = candidate.layout_info()
    assert len(info) == 6 and info[1:] == [128, 64, 128, 16, 2]
    emit("COMPILED_LAYOUT_CHECK", status="PASS", dynamic_smem_bytes=info[0],
         block_threads=info[1], state_rows=info[2], state_cols=info[3], chunk=info[4], parts=info[5],
         scope="actual_CuTe_bijections_transpose_TMA_and_sizeof; not_achieved_occupancy")
    emit("CANDIDATE_BINARY", file=candidate.__file__, sha256=sha256(candidate.__file__))

    naive_file = c1_root / "fla_kda_ref/naive.py"
    if sha256(naive_file) != FLA_NAIVE_SHA256:
        raise RuntimeError("FLA naive source hash mismatch")
    naive = runpy.run_path(str(naive_file))["naive_recurrent_kda"]
    emit("FLA_REFERENCE", file=str(naive_file), sha256=sha256(naive_file),
         precision="original_naive_internal_FP32; ordinary_sigmoid_normalization; not_FP64_gold")
    torch.manual_seed(20260910)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    result_file = build / "correctness.json"

    def save_record(tag, **fields):
        records.append(dict(tag=tag, **fields))
        result_file.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        emit(tag, **fields)

    def invoke(module, data, initial, want_final, state_dtype, cu):
        q, k, v, g, beta, alog, bias = data
        bsz, length, heads, _ = q.shape
        nseq = cu.numel() - 1 if cu is not None else bsz
        out = torch.full_like(v, float("nan"))
        final = torch.full((nseq, heads, 128, 128), float("nan"), device="cuda",
                           dtype=state_dtype) if want_final else None
        ws = torch.empty(module.get_workspace_size(bsz * length, heads, nseq),
                         dtype=torch.uint8, device="cuda")
        module.fwd(q, k, v, g, beta, 128 ** -0.5, out, ws, alog, bias, -5.0,
                   initial_state=initial, final_state=final, cu_seqlens=cu)
        torch.cuda.synchronize()  # 定位每次 launch/越界错误，不把异步错误带到后续样例。
        return out, final

    def compare(case, label, actual, expected):
        # 这里要求逐 bit 不增误差；不因失败而放宽 tolerance。FP32 API 也比较 bits。
        ints = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
        wrong = int((actual.view(ints) != expected.view(ints)).sum().item())
        nonfinite = int((~torch.isfinite(actual)).sum().item())
        baseline_nonfinite = int((~torch.isfinite(expected)).sum().item())
        save_record("BASELINE_COMPARE", case=case, tensor=label, elements=actual.numel(),
                    bit_mismatches=wrong, nonfinite=nonfinite, baseline_nonfinite=baseline_nonfinite,
                    max_abs=float((actual.float() - expected.float()).abs().max()),
                    status="PASS" if not (wrong or nonfinite or baseline_nonfinite) else "FAIL")
        if wrong or nonfinite or baseline_nonfinite:
            torch.save({"candidate": actual.cpu(), "baseline": expected.cpu()},
                       build / f"failure_{case}_{label}.pt")
            raise RuntimeError("baseline parity failed; saved arrays; no timing")

    def reference_metric(case, tensor, actual, reference):
        a, r = actual.detach().cpu().double(), reference.detach().cpu().double()
        error = a - r
        finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
        nrmse = float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(r).clamp_min(1e-30))
        save_record("FLA_COMPARE", case=case, tensor=tensor, nrmse=nrmse,
                    max_abs=float(error.abs().max()), all_finite=finite,
                    gate="candidate_bitwise_baseline; reference_error_identical_by_parity",
                    scope="short_fixture_FP32_naive; not_universal_accuracy_acceptance")
        if not finite:
            raise RuntimeError("nonfinite FLA comparison")

    def case(name, lengths, heads, batch, state_mode, reference=False, slow=False):
        is_varlen = lengths is not None
        length = sum(lengths) if is_varlen else (8192 if name.startswith("fixed_long") else 37)
        nseq = len(lengths) if is_varlen else batch
        shape = (batch, length, heads, 128)
        def normal():
            return torch.randn(shape, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        q, k = normal(), normal()
        # 列间不相同，尤其在 63/64 的分界；避免常数输入掩盖错误列偏移。
        column_scale = torch.linspace(0.25, 1.25, 128, device="cuda")
        v = (normal().float() * column_scale / 4).to(torch.bfloat16)
        g = torch.full(shape, -16.0 if slow else -2.0, device="cuda", dtype=torch.bfloat16)
        beta = torch.full(shape[:-1], -4.0 if slow else 0.0, device="cuda", dtype=torch.bfloat16)
        alog = torch.zeros(heads, device="cuda", dtype=torch.float32)
        bias = torch.zeros((heads, 128), device="cuda", dtype=torch.float32)
        dtype = torch.float32 if state_mode.endswith("f32") else torch.bfloat16
        has_in = state_mode.startswith("in") or state_mode.startswith("both")
        has_out = state_mode.startswith("out") or state_mode.startswith("both")
        initial = (torch.randn((nseq, heads, 128, 128), device="cuda") / 8).to(dtype) if has_in else None
        cu = None
        if is_varlen:
            prefix = [0]
            for n in lengths:
                prefix.append(prefix[-1] + n)
            cu = torch.tensor(prefix, device="cuda", dtype=torch.int64)
        emit("CASE_START", case=name, shape=shape, seq_lens=lengths, state=state_mode, slow=slow)
        data = (q, k, v, g, beta, alog, bias)
        saved_initial = initial.clone() if initial is not None else None
        try:
            base_o, base_s = invoke(baseline, data, initial, has_out, dtype, cu)
            # 基线必须没有改写只读初态；否则不能把被改过的初态给候选。
            if initial is not None:
                compare(name, "baseline_initial_unchanged", initial, saved_initial)
            cand_o, cand_s = invoke(candidate, data, initial, has_out, dtype, cu)
            compare(name, "output", cand_o, base_o)
            if has_out:
                compare(name, "state", cand_s, base_s)
            if initial is not None:
                compare(name, "candidate_initial_unchanged", initial, saved_initial)
        except Exception:
            # GPU context 若已报 illegal access，拷贝也可能失败；不得掩盖原始异常。
            try:
                torch.save({"q_k_v_g_beta_Alog_bias": [x.cpu() for x in data],
                            "initial_state": saved_initial.cpu() if has_in else None,
                            "seq_lens": lengths, "state_mode": state_mode},
                           build / f"failure_{name}_inputs.pt")
            except Exception as archive_error:
                emit("FAILURE_ARRAY_ARCHIVE", error=str(archive_error),
                     note="original_launch_or_parity_exception_follows")
            raise
        if reference:
            # FLA 接受激活后的输入与 [key,value] state；逐序列调用，避免跨序列传播。
            qf, kf = q.float().cpu(), k.float().cpu()
            qf *= torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            kf *= torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
            # 本 fixture 的 A_log/dt_bias 都是 0；不能把此简式用于任意模型输入。
            gf = -5.0 * torch.sigmoid(g.float().cpu())
            bf = torch.sigmoid(beta.float().cpu())
            vf = v.float().cpu()
            ref_o, ref_s = [], []
            bos = 0
            for seq in range(nseq):
                extent = lengths[seq] if is_varlen else length
                bid, start = (0, bos) if is_varlen else (seq, 0)
                sl = (slice(bid, bid + 1), slice(start, start + extent))
                si = saved_initial[seq:seq + 1].float().cpu().transpose(-1, -2) if has_in else None
                ro, rs = naive(qf[sl], kf[sl], vf[sl], gf[sl], bf[sl], scale=128 ** -0.5,
                               initial_state=si, output_final_state=True)
                ref_o.append(ro)
                ref_s.append(rs.transpose(-1, -2))
                bos += extent
            reference_metric(name, "output", cand_o, torch.cat(ref_o, dim=1 if is_varlen else 0))
            if has_out:
                reference_metric(name, "state", cand_s, torch.cat(ref_s, dim=0))

    # 7 种 state dispatch × fixed/varlen = 14 路；小形状用于定位布局/尾块/API 问题。
    modes = ["none", "in_bf16", "out_bf16", "both_bf16", "in_f32", "out_f32", "both_f32"]
    for variable in (False, True):
        for mode in modes:
            case(("varlen_" if variable else "fixed_") + mode,
                 [17, 33, 65] if variable else None, 4, 1, mode,
                 reference=mode in ("none", "both_bf16", "both_f32"))
    # 额外覆盖 batch 索引、题目 fixed 主形状、长链弱遗忘；此处仍没有性能计时。
    case("batch2_both_f32", None, 4, 2, "both_f32", reference=True)
    case("fixed_long_h96_both_bf16", None, 96, 1, "both_bf16")
    case("fixed_long_slow_h4", None, 4, 1, "both_bf16", slow=True)
    emit("CORRECTNESS_COMPLETED", status="PASS", cases=17, dispatch_variants=14,
         FLA_short_cases=7, scope="tested_full_KDA_cases_only; no_timing; no_universal_accuracy_claim")


class Tee:
    """同时写终端与原始日志；编译子进程的输出也进入同一文件。"""
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "cuda"), default="offline")
    parser.add_argument("--work", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout-seconds", type=int, default=1800,
                        help="编译+正确性子进程上限；超时只结束本脚本的进程组，保留日志")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        cuda_worker(args)
        return 0
    work = args.work.resolve()
    run = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    build = work / "build" / (run + "_task3_colsplit")
    raw = work / "logs/raw"
    raw.mkdir(parents=True, exist_ok=True)
    log = raw / (run + "_task3_colsplit.log")
    status = 1
    original_out, original_err = sys.stdout, sys.stderr
    with log.open("w", encoding="utf-8") as f:
        sys.stdout, sys.stderr = Tee(original_out, f), Tee(original_err, f)
        try:
            emit("PURPOSE", task="Task3_full_KDA_independent_value_column_split", mode=args.mode,
                 utc_run_id=run, script=str(Path(__file__).resolve()), python=sys.executable,
                 slurm_job_id=os.environ.get("SLURM_JOB_ID"), raw_log=str(log))
            build.mkdir(parents=True, exist_ok=False)
            # 每次保留当时运行器字节；后续改脚本也能核对旧清单中的哈希。
            (build / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
            originals, candidate = generate_candidate(work.parent, build / "csrc")
            offline_audit(originals, candidate)
            if args.mode == "cuda":
                if not os.environ.get("SLURM_JOB_ID"):
                    raise RuntimeError("run inside your existing Slurm GPU allocation")
                if not Path("/usr/local/cuda/bin/nvcc").is_file():
                    raise RuntimeError("missing /usr/local/cuda/bin/nvcc")
                env = dict(os.environ)
                # 使用 sys.prefix 保留虚拟环境身份，避免解析 python 符号链接后落到系统 bin。
                # 只写传给 Popen 的 env；父 shell 与当前 Python 的 PATH 不受影响。
                venv_bin = os.path.join(sys.prefix, "bin")
                env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
                env["CUDA_HOME"] = "/usr/local/cuda"
                env["MAX_JOBS"] = "2"
                # 复用已准备的 Python headers，不下载环境；保留已有 CPATH 条目。
                headers = sorted(work.glob("python/cpython-*/include/python3.12/Python.h"))
                if headers:
                    env["CPATH"] = str(headers[0].parent) + (":" + env["CPATH"] if env.get("CPATH") else "")
                emit("BUILD_ENV", python_include=str(headers[0].parent) if headers else "Python_default",
                     timeout_seconds=args.timeout_seconds, venv_bin=venv_bin,
                     ninja_in_worker_path=shutil.which("ninja", path=env["PATH"]))
                snapshot(raw / (run + "_task3_colsplit_gpu_before.csv"))
                cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "cuda",
                       "--work", str(work), "--worker", str(build)]
                emit("COMMAND", argv=cmd)
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, env=env, start_new_session=True)
                timed_out = threading.Event()
                def terminate_worker():
                    # 仅终止此脚本启动的独立进程组；不触碰 Slurm 或其他 GPU 进程。
                    timed_out.set()
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                timer = threading.Timer(args.timeout_seconds, terminate_worker)
                timer.start()
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                    status = process.wait()
                finally:
                    timer.cancel()
                if timed_out.is_set():
                    emit("TIMEOUT", seconds=args.timeout_seconds, status="FAIL_no_timing")
                emit("COMMAND_EXIT", exit=status)
                snapshot(raw / (run + "_task3_colsplit_gpu_after.csv"))
                if status:
                    raise RuntimeError(f"candidate compile/correctness worker failed: {status}")
            else:
                emit("OFFLINE_COMPLETED", status="PASS", candidate_sources=str(build / "csrc"),
                     scope="generated_and_audited_only; no_CUDA_compile_or_GPU_execution")
            status = 0
        except Exception:
            traceback.print_exc()
            status = 1
        finally:
            emit("PROBE_EXIT", exit=status, performance="not_measured")
            sys.stdout, sys.stderr = original_out, original_err
    # 清单在 log 关闭后生成，避免把自己的哈希写回本体；失败构建的产物也保留。
    files = [Path(__file__).resolve(), log]
    files += sorted(build.rglob("*")) if build.exists() else []
    files += sorted(raw.glob(run + "_task3_colsplit_gpu_*.csv"))
    files = [p for p in files if p.is_file()]
    manifest = Path(str(log) + ".sha256")
    manifest.write_text("".join(f"{sha256(p)}  {p}\n" for p in files))
    for p in files:
        if p.suffix in (".py", ".log", ".json", ".patch", ".so", ".csv"):
            print(f"{sha256(p)}  {p}")
    print(f"raw_log={log}\nhash_manifest={manifest}\ncandidate_patch={build / 'candidate.patch'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
