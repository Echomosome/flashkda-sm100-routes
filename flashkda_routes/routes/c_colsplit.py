"""Route C: split the value dimension across two CTAs per head.

This route changes *task parallelism*, not the algorithm: the mathematical
``[key, value]`` state is cut into two independent 64-column value shards, so
the recurrence kernel gets twice as many CTAs and each CTA owns half the
state.  The key dimension stays 128 and every reduction axis is preserved.

The patch is expressed as exact text substitutions against the pinned FlashKDA
sources.  Every anchor asserts its hit count; a changed upstream file stops the
generation instead of silently producing a different kernel.
"""

import difflib
import json
from pathlib import Path

from ..config import SOURCE_SHA256, sha256, sha256_text
from ..emit import emit


def replace_exact(text, old, new, count=1):
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"patch anchor count {actual} != {count}: {old[:140]!r}")
    return text.replace(old, new)


LAYOUT_CHECK_CUDA = r'''
// Host-side enumeration of the real CuTe layouts: range, uniqueness,
// transpose aliasing and TMA leading dimension.  This check only exists after
// nvcc compiles it; a Python address model is not a substitute.
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
    """Copy the extension and rewrite K2's V/O/state to half width.

    K1, CHUNK, the key dimension and every floating-point expression inside a
    phase stay unchanged.  Only column ownership, layouts and thread roles move.
    Returns the original and revised source dictionaries for auditing.
    """
    source = Path(c1_root) / "FlashKDA/csrc"
    originals = {}
    for name, expected in SOURCE_SHA256.items():
        path = source / name
        if sha256_text(path) != expected:
            raise RuntimeError(f"baseline source hash mismatch: {path}")
        originals[name] = path.read_text(encoding="utf-8")
        emit("BASELINE_SOURCE", file=str(path), sha256=expected)

    candidate = dict(originals)
    k2 = candidate["smxx/fwd_kernel2.cuh"]
    k2 = replace_exact(k2, '#include "utils.cuh"',
                       '#include "utils.cuh"\n#include "task3_state_cast.cuh"')
    k2 = replace_exact(k2, 'struct K2Layouts {', '''struct K2Layouts {
    // Global K=V=128; this CTA keeps only DV=64 value columns, key stays D.
    static_assert(D == 128 && CHUNK == 16, "Task3 v1 only supports D128/C16");
    static constexpr int DV = D / 2;''')
    k2 = replace_exact(k2, '''    using VOLayout = MMALayout;
    using TransposedVOLayout = TransposedMMALayout;''', '''    // V/O are [CHUNK, DV]; KD/QD/KR keep the full [CHUNK, D].
    using VOLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<DV>{}), LayoutLeft{}));
    using TransposedVOLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<DV>{}, Int<CHUNK>{}), LayoutRight{}));''')
    # Three state views: [DV,D], transposed [D,DV], FP32 [DV,D].
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
    k2 = replace_exact(k2, "    using TMAStateSmemLayout = decltype(composition(", '''    // Keep a full-workspace TMA view; it must not inherit the shrunken VO view.
    using TMAMMALayout = decltype(composition(
        MMALayout{}.layout_a(), MMALayout{}.offset(), prepend(MMALayout{}.layout_b())
    ));
    using TMAStateSmemLayout = decltype(composition(''')
    k2 = replace_exact(k2, "    using TMAVOLayout = typename Layouts::TMAVOLayout;",
                       "    using TMAVOLayout = typename Layouts::TMAVOLayout;\n"
                       "    using TMAMMALayout = typename Layouts::TMAMMALayout;\n"
                       "    constexpr int DV = Layouts::DV;")
    k2 = replace_exact(k2, "constexpr int kComputeThreads = 128;", '''constexpr int kComputeThreads = DV;
    // One warp still owns 32 value columns: 2 compute + 1 load + 1 store.
    static_assert(NumThreads == kComputeThreads + 64, "warp roles mismatch");''')
    k2 = replace_exact(k2, "    int head_idx = blockIdx.y;",
                       "    int head_idx = blockIdx.y;\n"
                       "    int value_offset = int(blockIdx.z) * DV;  // shard 0/1: columns 0/64")
    k2 = replace_exact(k2, "(seq_idx * H + head_idx, 0, 0)",
                       "(seq_idx * H + head_idx, value_offset, 0)", 4)
    k2 = replace_exact(k2, "make_shape(Int<1>{}, Int<D>{}, Int<D>{})",
                       "make_shape(Int<1>{}, Int<DV>{}, Int<D>{})", 4)
    for tensor in ("v", "out"):
        k2 = replace_exact(k2, f"g_{tensor}.layout()(head_idx, int(bos) + t * CHUNK, 0)",
                           f"g_{tensor}.layout()(head_idx, int(bos) + t * CHUNK, value_offset)")
        k2 = replace_exact(
            k2,
            f"make_shape(Int<1>{{}}, Int<CHUNK>{{}}, Int<D>{{}}), stride(g_{tensor}.layout())",
            f"make_shape(Int<1>{{}}, Int<CHUNK>{{}}, Int<DV>{{}}), stride(g_{tensor}.layout())")
    for tensor in ("k_decayed", "q_decayed", "k_restored"):
        k2 = replace_exact(k2, f"{tensor}.begin()), TMAVOLayout{{}})",
                           f"{tensor}.begin()), TMAMMALayout{{}})")
    k2 = replace_exact(k2, "* H * D + head_idx * D;", "* H * D + head_idx * D + value_offset;")
    k2 = replace_exact(k2, "col < D;", "col < DV;")
    k2 = replace_exact(k2, "// Only one thread (lane_predicate) runs here, so loop over all D",
                       "// Task3: the store leader writes this shard's DV columns; the tail\n"
                       "// block must not spill into the next sequence.")
    k2 = replace_exact(k2, "smem_cvt_fp32_to_bf16<FP32StateSmemLayout, StateSmemLayout, D, NumThreads>",
                       "c1_rect_fp32_to_bf16<FP32StateSmemLayout, StateSmemLayout, DV, D, NumThreads>")
    k2 = replace_exact(k2, "smem_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, D, NumThreads>",
                       "c1_rect_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, DV, D, NumThreads>")
    k2 = k2.replace("_flash_kda_fwd_recurrence", "_c1_colsplit_fwd_recurrence")
    candidate["smxx/fwd_kernel2.cuh"] = k2

    # Reuse the original 8x8 cast atoms, but iterate a rectangular Rows x Cols
    # region.  utils.cuh itself is untouched.
    utils = originals["smxx/utils.cuh"]
    casts = utils[utils.index(
        "template <class FP32Layout, class BF16Layout, int D, int NumThreads>"):]
    casts = casts.replace("int D, int NumThreads", "int Rows, int Cols, int NumThreads")
    casts = replace_exact(
        casts, "constexpr int kBlocksPerDim = D / kBlock;",
        "static_assert(Rows % 8 == 0 && Cols % 8 == 0, \"8x8 cast tiles\");\n"
        "    constexpr int kBlocksPerDim = Cols / kBlock;", 2)
    casts = replace_exact(
        casts, "constexpr int kTotalBlocks = kBlocksPerDim * kBlocksPerDim;",
        "constexpr int kTotalBlocks = (Rows / kBlock) * kBlocksPerDim;", 2)
    casts = casts.replace("smem_cvt_fp32_to_bf16", "c1_rect_fp32_to_bf16")
    casts = casts.replace("smem_cvt_bf16_to_fp32", "c1_rect_bf16_to_fp32")
    candidate["smxx/task3_state_cast.cuh"] = (
        '#pragma once\n#include "utils.cuh"\n'
        '// Task3: physical state is [value,key]; this CTA has Rows=64, Cols=128.\n'
        '// Keep the original BF16 arithmetic and 8x8 atoms; fix the square helper\n'
        '// so it covers the rectangular tile count.\n' + casts)

    launch = candidate["smxx/fwd_launch.cu"]
    launch = replace_exact(launch, '#include "fwd.h"', '''#include "fwd.h"
#include <vector>
#include <stdexcept>
// K1 algorithm and source are unchanged; only symbols are renamed so this
// extension cannot be confused with the installed baseline DSO.
#define _flash_kda_fwd_prepare _c1_colsplit_fwd_prepare
#define _flash_kda_build_tile_prefix _c1_colsplit_build_tile_prefix''')
    launch = replace_exact(launch, "    // TMA layouts for Kernel 2",
                           "    // Task3: only raw V and output O use the half-width TMA;\n"
                           "    // the workspace stays full-width.\n"
                           "    using K2TMAVOLayout = typename K2L::TMAVOLayout;\n"
                           "    // TMA layouts for Kernel 2")
    for tensor in ("v", "out"):
        launch = replace_exact(launch, f"m_{tensor}, TMAVOLayout{{}})",
                               f"m_{tensor}, K2TMAVOLayout{{}})")
    launch = replace_exact(launch, "constexpr int kK2Threads = 32 * 2 + 128;",
                           "constexpr int kK2Threads = 32 * 2 + K2L::DV;")
    launch = replace_exact(launch, "dim3 grid_k2(N, H);",
                           "dim3 grid_k2(N, H, 2);  // two independent value shards")
    launch = launch.replace("_flash_kda_fwd_recurrence", "_c1_colsplit_fwd_recurrence")
    launch += LAYOUT_CHECK_CUDA
    candidate["smxx/fwd_launch.cu"] = launch

    # Isolate C++ host symbols too; the Python API names stay fwd/get_workspace_size.
    for name in ("fwd.h", "flash_kda.cpp", "smxx/fwd_launch.cu"):
        candidate[name] = __import__("re").sub(r"\blaunch_fwd\b", "c1_colsplit_launch_fwd", candidate[name])
    cpp = candidate["flash_kda.cpp"]
    cpp = replace_exact(cpp, "#include <torch/extension.h>",
                        "#include <torch/extension.h>\n#include <pybind11/stl.h>\n#include <vector>")
    cpp = replace_exact(cpp, "void fwd(", "void c1_colsplit_fwd(")
    cpp = replace_exact(cpp, 'm.def("fwd", &fwd,', 'm.def("fwd", &c1_colsplit_fwd,')
    cpp = replace_exact(cpp, "int64_t get_workspace_size(", "int64_t c1_colsplit_get_workspace_size(")
    cpp = replace_exact(cpp, "(&get_workspace_size)", "(&c1_colsplit_get_workspace_size)")
    cpp = replace_exact(cpp, "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", '''// Host-side check of the real CuTe layouts; returns sizeof of the compiled
// shared-storage struct.  It does not run a GPU kernel.
std::vector<int64_t> c1_colsplit_layout_info();
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layout_info", &c1_colsplit_layout_info);''')
    candidate["flash_kda.cpp"] = cpp

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    patch = []
    for name, revised in candidate.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(revised, encoding="utf-8")
        patch.extend(difflib.unified_diff(
            originals.get(name, "").splitlines(True), revised.splitlines(True),
            fromfile="baseline/csrc/" + name, tofile="candidate/csrc/" + name))
    (destination.parent / "candidate.patch").write_text("".join(patch), encoding="utf-8")
    (destination.parent / "source_manifest.json").write_text(json.dumps({
        "baseline": SOURCE_SHA256,
        "candidate": {name: sha256(destination / name) for name in candidate},
        "scope": "full_KDA_parallelism_candidate; CUDA_not_compiled_by_generation",
    }, indent=2) + "\n", encoding="utf-8")
    return originals, candidate


def offline_audit(originals, candidate):
    """Source invariants, logical address ownership and column separability.

    This is an offline check.  It does not compile CUDA and cannot replace a
    CuTe layout check, a TMA test or a GPU comparison.
    """
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

    # The two CTAs' global state addresses do not overlap and cover one head;
    # the FP32 cast covers the shard exactly once.
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

    # The state update for value shard p only needs that shard: key reduction
    # stays complete, so no cross-shard state reduction is required.
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
