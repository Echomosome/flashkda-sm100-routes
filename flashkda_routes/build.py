"""Generate isolated candidate sources and compile them as Torch JIT extensions.

Nothing here installs a package or writes into the FlashKDA checkout.  Each
route gets its own source directory, patch and binary under the run directory.
"""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

from .config import (
    SOURCE_SHA256, arch_flags, cuobjdump, cutlass_include, flashkda_csrc,
    sha256, sha256_text, upstream_root,
)
from .emit import emit
from .routes import c_colsplit


def _a_patch_source(work, dest, route):
    """Write Route A / A_control sources and return the candidate directory."""
    work, dest = Path(work), Path(dest)
    originals = {}
    for name, digest in SOURCE_SHA256.items():
        path = flashkda_csrc(work) / name
        if sha256_text(path) != digest:
            raise RuntimeError(f"baseline source changed: {path}")
        originals[name] = path.read_text(encoding="utf-8")

    candidate = dict(originals)
    replace = c_colsplit.replace_exact
    k2 = candidate["smxx/fwd_kernel2.cuh"]
    k2 = replace(k2, '#include "utils.cuh"', '#include "utils.cuh"\n#include "c1_phase6.cuh"')
    k2 = replace(
        k2,
        "    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;\n",
        "    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;\n"
        "    c1_phase6::Storage<BF16> phase6;  // one TMEM allocation per CTA for the whole chain\n")
    k2 = replace(
        k2,
        "        int compute_tid = threadIdx.x;\n",
        "        int compute_tid = threadIdx.x;\n"
        '        static_assert(D == 128 && CHUNK == 16, "A route scope");\n'
        "        c1_phase6::init(shared_storage.phase6, compute_barrier, compute_tid);\n")
    start = k2.index("            // ======== Phase 6: s_acc update ========")
    end = k2.index("            }\n            compute_barrier.arrive_and_wait();", start)
    k2 = k2[:start] + """            // ======== Phase 6: actual U -> packed inputs -> MMA -> state ========
            Tensor p6_u = make_tensor(make_smem_ptr(shared_storage.phase6.u), VOLayout{});
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor u_block = local_tile(p6_u, make_shape(Int<16>{}, Int<16>{}),
                                            make_coord(0, warp_id * 2 + i));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(u_bf16[i]),
                     smem_thr_store_C.partition_D(u_block));
            }
            c1_phase6::update(shared_storage.phase6, p6_u, k_restored, s_acc_T,
                              g_total, compute_barrier, compute_tid, t);
""" + k2[end:]
    end_loop = """            ++out_write;
#endif
        }
    }
"""
    k2 = replace(k2, end_loop, """            ++out_write;
#endif
        }
        c1_phase6::finish(shared_storage.phase6, compute_tid);
    }
""")
    k2 = replace(k2, "_flash_kda_fwd_recurrence(", "_c1_phase6_fwd_recurrence(")
    candidate["smxx/fwd_kernel2.cuh"] = k2

    launch = replace(candidate["smxx/fwd_launch.cu"], "_flash_kda_fwd_recurrence<",
                     "_c1_phase6_fwd_recurrence<")
    launch += """
// sizeof is the compiler's actual budget; it is not achieved occupancy.
int64_t c1_phase6_smem_bytes() {
    return sizeof(SharedStorageK2<K2Layouts<128,16>,3,2>);
}
"""
    candidate["smxx/fwd_launch.cu"] = launch

    cpp = candidate["flash_kda.cpp"]
    candidate["flash_kda.cpp"] = replace(
        cpp, "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {",
        "int64_t c1_phase6_smem_bytes();\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        '    m.def("smem_bytes", &c1_phase6_smem_bytes);')

    header = (Path(__file__).parent / "routes" / "a_phase6.cuh").read_text(encoding="utf-8")
    if route == "A_control":
        header = "#define C1_PHASE6_HMMA_CONTROL 1\n" + header
    candidate["smxx/c1_phase6.cuh"] = header

    # Offline invariant: K1, normalisation, workspace and every prior floating
    # point expression in K2 are untouched.
    for name in ("smxx/fwd_kernel1.cuh", "smxx/utils.cuh", "fwd.h"):
        assert candidate[name] == originals[name]
    begin = "            // Fused MMA:"
    stop = "            // ======== Phase 6:"
    original_k2 = originals["smxx/fwd_kernel2.cuh"]
    assert (k2[k2.index(begin):k2.index(stop)] ==
            original_k2[original_k2.index(begin):original_k2.index(stop)])

    import difflib
    diff = []
    for name, value in candidate.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        diff.extend(difflib.unified_diff(
            originals.get(name, "").splitlines(True), value.splitlines(True),
            fromfile="baseline/csrc/" + name, tofile="candidate/csrc/" + name))
    (dest.parent / "candidate.patch").write_text("".join(diff), encoding="utf-8")
    (dest.parent / "source_manifest.json").write_text(json.dumps({
        "route": route,
        "baseline": SOURCE_SHA256,
        "candidate": {name: sha256(dest / name) for name in candidate},
        "status": "OFFLINE_GENERATED; CUDA_compile_and_full_KDA_correctness_pending",
    }, indent=2) + "\n", encoding="utf-8")
    return dest


def generate(work, build_dir, route):
    """Generate the candidate source tree for ``route`` under ``build_dir``."""
    build_dir = Path(build_dir)
    dest = build_dir / "csrc"
    if route == "C":
        originals, candidate = c_colsplit.generate_candidate(upstream_root(work), dest)
        c_colsplit.offline_audit(originals, candidate)
        return dest
    if route in ("A", "A_control"):
        return _a_patch_source(work, dest, route)
    raise ValueError(f"unknown native route: {route}")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _count_sass(module_path, route, route_dir):
    """Dump SASS and count MMA opcodes per function; returns the counts or None."""
    tool = cuobjdump()
    if not Path(tool).is_file():
        emit("SASS_SKIPPED", reason="cuobjdump not found", tool=str(tool))
        return None
    sass = route_dir / "sass.txt"
    with sass.open("w") as f:
        subprocess.run([str(tool), "--dump-sass", module_path],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    functions, current = {}, None
    for line in sass.read_text(encoding="utf-8").splitlines():
        m = re.search(r"Function\s*:\s*(\S+)", line)
        if m:
            current = m.group(1)
            functions.setdefault(current, {})
        m = re.search(r"/\*[0-9a-fA-F]+\*/\s*(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b", line)
        if m and current:
            opcode = m.group(1)
            if "MMA" in opcode:
                functions[current][opcode] = functions[current].get(opcode, 0) + 1
    (route_dir / "sass_counts.json").write_text(json.dumps(functions, indent=2) + "\n", encoding="utf-8")
    selected = {n: v for n, v in functions.items() if "recurrence" in n}
    emit("SASS_RECURRENCE", route=route, functions=selected, counts="static_not_dynamic")
    if not selected:
        raise RuntimeError("recurrence symbol missing from SASS")
    if route == "A" and any(not any(op.startswith("UTCHMMA") for op in ops)
                            for ops in selected.values()):
        raise RuntimeError("tcgen05 SASS missing in a recurrence variant")
    if route != "A" and any(any(op.startswith("UTCHMMA") for op in ops)
                            for ops in selected.values()):
        raise RuntimeError("unexpected UTCHMMA in HMMA control")
    return selected


def build_native(work, route_dir, route, arch="10.3a"):
    """Compile one candidate as an isolated Torch extension and return the module."""
    from torch.utils.cpp_extension import load

    work, route_dir = Path(work), Path(route_dir)
    source = generate(work, route_dir, route)
    cutlass = cutlass_include(work)
    if not (cutlass / "cute" / "tensor.hpp").is_file():
        raise RuntimeError(f"pinned CUTLASS headers missing under {cutlass}; run scripts/fetch_upstream.sh")

    jit = route_dir / "jit"
    jit.mkdir()
    name = "flashkda_routes_" + route.lower() + "_" + re.sub("[^0-9]", "", route_dir.parent.name)
    flags = ["-O3", *arch_flags(arch),
             "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
             "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
             "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills",
             "-lineinfo", "--threads", "2", "-Xcompiler=-fvisibility=hidden"]
    emit("BUILD", route=route, flags=flags, mode="isolated_JIT_no_install")
    module = load(
        name=name,
        sources=[str(source / "flash_kda.cpp"), str(source / "smxx/fwd_launch.cu")],
        extra_include_paths=[str(source), str(source / "smxx"), str(cutlass),
                             str(cutlass.parent / "examples/common"),
                             str(cutlass.parent / "tools/util/include")],
        extra_cflags=["-O3", "-Wno-psabi", "-fvisibility=hidden"],
        extra_cuda_cflags=flags,
        build_directory=str(jit), verbose=True)

    identity = dict(name=module.__name__, path=str(Path(module.__file__).resolve()),
                    sha256=sha256(module.__file__))
    (route_dir / "binary.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    emit("BINARY", route=route, **identity)
    if route == "C":
        info = module.layout_info()
        if list(info[1:]) != [128, 64, 128, 16, 2]:
            raise RuntimeError("Route C layout check failed")
        emit("COMPILED_LAYOUT", route=route, values=list(info))
    else:
        emit("COMPILED_SMEM", route=route, bytes=module.smem_bytes(), threads=192, chunk=16,
             extra_buffers="U_and_packed_A_B; actual_TMEM_residency_not_measured")
    _count_sass(module.__file__, route, route_dir)
    return module
