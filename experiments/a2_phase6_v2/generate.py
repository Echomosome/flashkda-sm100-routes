"""生成独立源码目录及 diff；只写本轮 build，不修改官方 FlashKDA。

A2：直接消费 MN_INTER 的 KR/U，按 TMEM warp 行范围更新状态。
A2_control：同一 U 落盘、warp 分工和 epilogue，仍用 HMMA。
只生成 A/A_control；旧 v1 和 C 路线的代码、实测记录保持原样。
"""
from pathlib import Path
import difflib
import hashlib
import importlib.util
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_generator(work):
    # 随新增目录交付同一生成器的原字节快照，避免远端脚本名/PATH 修复版本不同。
    path = Path(__file__).with_name('p2_generator.py')
    spec = importlib.util.spec_from_file_location('c1_existing_p2', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate(work, build, route):
    work, build = Path(work), Path(build)
    old = old_generator(work)
    dest = build / 'csrc'
    if route not in ('A', 'A_control'):
        raise ValueError(route)
    originals = {}
    for name, digest in old.SOURCE_SHA256.items():
        path = work.parent / 'FlashKDA/csrc' / name
        if sha(path) != digest:
            raise RuntimeError(f'baseline source changed: {path}')
        originals[name] = path.read_text()
    candidate = dict(originals)
    replace = old.replace_exact
    k2 = candidate['smxx/fwd_kernel2.cuh']
    k2 = replace(k2, '#include "utils.cuh"', '#include "utils.cuh"\n#include "c1_phase6.cuh"')
    k2 = replace(k2, '    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;\n',
                 '    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;\n'
                 '    c1_phase6::Storage<BF16> phase6;  // 本 CTA 全链只分配一次 TMEM\n')
    k2 = replace(k2, '        int compute_tid = threadIdx.x;\n',
                 '        int compute_tid = threadIdx.x;\n'
                 '        static_assert(D == 128 && CHUNK == 16, "A route scope");\n'
                 '        c1_phase6::init(shared_storage.phase6, compute_barrier, compute_tid);\n')
    start = k2.index('            // ======== Phase 6: s_acc update ========')
    end = k2.index('            }\n            compute_barrier.arrive_and_wait();', start)
    body = Path(__file__).with_name('phase6_body.inc').read_text()
    k2 = k2[:start] + body + k2[end:]
    end_loop = '''            ++out_write;
#endif
        }
    }
'''
    k2 = replace(k2, end_loop, '''            ++out_write;
#endif
        }
        c1_phase6::finish(shared_storage.phase6, compute_tid);
    }
''')
    k2 = replace(k2, '_flash_kda_fwd_recurrence(', '_c1_phase6_fwd_recurrence(')
    candidate['smxx/fwd_kernel2.cuh'] = k2
    launch = replace(candidate['smxx/fwd_launch.cu'], '_flash_kda_fwd_recurrence<', '_c1_phase6_fwd_recurrence<')
    launch += '''
// sizeof 是编译器实际预算；不当作 achieved occupancy。
int64_t c1_phase6_smem_bytes() {
    return sizeof(SharedStorageK2<K2Layouts<128,16>,3,2>);
}
// 在 CPU 上调用 CuTe 原始 layout；先核验全部坐标，再允许执行候选 kernel。
int64_t c1_phase6_layout_errors() {
    using L = K2Layouts<128,16>;
    int64_t errors = 0;
    for (int m=0; m<128; ++m) for (int k=0; k<16; ++k) {
        int expected = m%8 + (k%8)*8 + (m/8)*128 + (k/8)*64;
        errors += int(L::TransposedMMALayout{}(make_coord(m,k))) != expected;
        errors += int(L::VOLayout{}(make_coord(k,m))) != expected;
    }
    // 核对 TMEM .16x256b.x2 的寄存器顺序与真实 CuTe C 片段，而不仅是手写整数模型。
    auto mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                             Layout<Shape<_1,_1>>{}, Tile<_16,_16,_16>{});
    auto identity = make_identity_tensor(make_shape(Int<16>{},Int<16>{}));
    for (int lane=0; lane<32; ++lane) {
        auto coords = mma.get_slice(lane).partition_C(identity);
        for (int d=0; d<2; ++d) for (int b=0; b<2; ++b) for (int a=0; a<2; ++a) {
            auto pos = coords(a+2*b+4*d);
            errors += int(get<0>(pos)) != lane/4+8*b;
            errors += int(get<1>(pos)) != 2*(lane%4)+a+8*d;
        }
    }
    return errors;
}
'''
    candidate['smxx/fwd_launch.cu'] = launch
    cpp = candidate['flash_kda.cpp']
    candidate['flash_kda.cpp'] = replace(cpp, 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {',
        'int64_t c1_phase6_layout_errors();\nint64_t c1_phase6_smem_bytes();\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
        '    m.def("smem_bytes", &c1_phase6_smem_bytes);\n'
        '    m.def("layout_errors", &c1_phase6_layout_errors);')
    header = Path(__file__).with_name('phase6.cuh').read_text()
    if route == 'A_control':
        header = '#define C1_PHASE6_HMMA_CONTROL 1\n' + header
    candidate['smxx/c1_phase6.cuh'] = header
    # 离线不变量：K1、归一化、workspace、所有前序浮点表达式未被替换。
    for name in ('smxx/fwd_kernel1.cuh', 'smxx/utils.cuh', 'fwd.h'):
        assert candidate[name] == originals[name]
    begin = '            // Fused MMA:'
    stop = '            // ======== Phase 6:'
    original_k2 = originals['smxx/fwd_kernel2.cuh']
    assert k2[k2.index(begin):k2.index(stop)] == original_k2[original_k2.index(begin):original_k2.index(stop)]
    diff = []
    for name, value in candidate.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        diff.extend(difflib.unified_diff(originals.get(name, '').splitlines(True), value.splitlines(True),
                    fromfile='baseline/csrc/' + name, tofile='candidate/csrc/' + name))
    (build / 'candidate.patch').write_text(''.join(diff))
    (build / 'source_manifest.json').write_text(json.dumps({
        'route': route, 'baseline': old.SOURCE_SHA256,
        'candidate': {name:sha(dest / name) for name in candidate},
        'status':'OFFLINE_GENERATED; CUDA_compile_and_full_KDA_correctness_pending'
    }, indent=2) + '\n')
    return dest
