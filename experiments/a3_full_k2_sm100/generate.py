"""A3 生成器：固定 baseline 的完整独立副本，替换 K2 compute body。

不会改官方仓库或 A2 已测代码。每处锚点和 baseline SHA 先核对，避免静默漏改。
"""
from pathlib import Path
import difflib
import hashlib
import importlib.util
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_generator(work=None):
    p=Path(__file__).with_name('source_pins.py')
    spec=importlib.util.spec_from_file_location('c1_pinned_source_info',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def wide_load():
    """PTX 的寄存器列表必须静态展开；数组下标和输出个数仍可审查。"""
    registers=','.join('%'+str(i) for i in range(64))
    operands=','.join('"=f"(f['+str(i)+'])' for i in range(64))
    return '\nnamespace c1_transpose {\n' + '''
__device__ __forceinline__ void load128(uint32_t addr,float (&f)[64]) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x16.b32 {REGS},[%64];"
                 :OPERANDS:"r"(addr):"memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
}
} // namespace c1_transpose
'''.replace('REGS',registers).replace('OPERANDS',operands)


LAYOUT_CHECK = r'''
// 编译后在 host 执行真实 CuTe 布局枚举；不等价于 GPU 对拍。
int64_t c1_transpose_layout_errors() {
    using L=K2Layouts<128,16>;
    using U=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
                    make_shape(Int<128>{},Int<16>{}),LayoutLeft{}));
    using KQ=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
                    make_shape(Int<32>{},Int<128>{}),LayoutLeft{}));
    int64_t errors=0;
    auto k_offset=[](int m,int k,int rows){return k%8+8*(m%8)+64*(m/8)+8*rows*(k/8);};
    for(int m=0;m<128;++m) for(int k=0;k<128;++k)
        errors += int(L::StateSmemLayout{}(make_coord(m,k)))!=k_offset(m,k,128);
    for(int m=0;m<128;++m) for(int k=0;k<16;++k) {
        errors += int(U{}(make_coord(m,k)))!=k_offset(m,k,128);
        errors += int(L::TransposedMMALayout{}(make_coord(m,k)))!=m%8+8*(k%8)+128*(m/8)+64*(k/8);
        errors += int(L::TransposedVOLayout{}(make_coord(m,k)))!=int(L::VOLayout{}(make_coord(k,m)));
    }
    for(int m=0;m<16;++m) for(int k=0;k<128;++k)
        errors += int(L::MMALayout{}(make_coord(m,k)))!=k_offset(m,k,16);
    for(int m=0;m<16;++m) for(int k=0;k<16;++k)
        errors += int(L::LMLayout{}(make_coord(m,k)))!=k_offset(m,k,16);
    for(int m=0;m<32;++m) for(int k=0;k<128;++k)
        errors += int(KQ{}(make_coord(m,k)))!=k_offset(m,k,32);
    auto tiled=make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                             Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{});
    auto id=make_identity_tensor(make_shape(Int<16>{},Int<16>{}));
    for(int lane=0;lane<32;++lane) {
        auto c=tiled.get_slice(lane).partition_C(id);
        for(int j=0;j<8;++j) {
            errors += int(get<0>(c(j)))!=lane/4+8*((j/2)%2);
            errors += int(get<1>(c(j)))!=2*(lane%4)+(j%2)+8*(j/4);
        }
    }
    return errors;
}
int64_t c1_transpose_smem_bytes() {
    return sizeof(SharedStorageK2<K2Layouts<128,16>,3,2>);
}
'''


def generate(work,build,route):
    work,build=Path(work),Path(build)
    if route not in ('A','A_control'): raise ValueError(route)
    old=old_generator();replace=old.replace_exact
    originals={}
    for name,digest in old.SOURCE_SHA256.items():
        p=work.parent/'FlashKDA/csrc'/name
        if sha(p)!=digest: raise RuntimeError(f'baseline hash mismatch: {p}')
        originals[name]=p.read_text()
    candidate=dict(originals);k2=candidate['smxx/fwd_kernel2.cuh']
    k2=replace(k2,'#include "utils.cuh"','#include "utils.cuh"\n#include "c1_sm100_ops.cuh"')
    k2=replace(k2,'    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;\n',
        '    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;\n'
        '    c1_transpose::Storage<BF16> transposed;\n')
    k2=replace(k2,'        int compute_tid = threadIdx.x;\n',
        '        int compute_tid = threadIdx.x;\n'
        '        static_assert(D==128 && CHUNK==16,"A3 scope");\n'
        '        int completion_phase=0;  // 每 chunk 三次 commit，不能直接用 chunk&1。\n'
        '        c1_transpose::init(shared_storage.transposed,compute_barrier,compute_tid);\n')
    start=k2.index('            // Fused MMA:')
    end=k2.index('            compute_barrier.arrive_and_wait();',start)
    k2=k2[:start]+Path(__file__).with_name('recurrence_body.inc').read_text()+k2[end:]
    tail='''            ++out_write;
#endif
        }
    }
'''
    k2=replace(k2,tail,'''            ++out_write;
#endif
        }
        c1_transpose::finish(shared_storage.transposed,compute_tid);
    }
''')
    k2=replace(k2,'_flash_kda_fwd_recurrence(','_c1_sm100_transpose_recurrence(')
    candidate['smxx/fwd_kernel2.cuh']=k2
    candidate['smxx/fwd_launch.cu']=replace(candidate['smxx/fwd_launch.cu'],
        '_flash_kda_fwd_recurrence<','_c1_sm100_transpose_recurrence<')+LAYOUT_CHECK
    candidate['flash_kda.cpp']=replace(candidate['flash_kda.cpp'],
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {',
        'int64_t c1_transpose_layout_errors();\nint64_t c1_transpose_smem_bytes();\n'
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
        '    m.def("layout_errors",&c1_transpose_layout_errors);\n'
        '    m.def("smem_bytes",&c1_transpose_smem_bytes);')
    header=Path(__file__).with_name('sm100_ops.cuh').read_text()+wide_load()
    if route=='A_control': header='#define C1_TRANSPOSE_HMMA_CONTROL 1\n'+header
    candidate['smxx/c1_sm100_ops.cuh']=header
    for name in ['fwd.h','smxx/utils.cuh','smxx/fwd_kernel1.cuh']:
        assert candidate[name]==originals[name]
    dest=build/'csrc';diff=[]
    for name,data in candidate.items():
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
        diff.extend(difflib.unified_diff(originals.get(name,'').splitlines(True),data.splitlines(True),
                    fromfile='baseline/csrc/'+name,tofile='candidate/csrc/'+name))
    (build/'candidate.patch').write_text(''.join(diff))
    (build/'source_manifest.json').write_text(json.dumps(dict(route=route,generation='A3_all_K2_transposed',
        baseline=old.SOURCE_SHA256,candidate={n:sha(dest/n) for n in candidate},
        correctness='CUDA_PENDING',performance='NOT_MEASURED'),indent=2)+'\n')
    return dest
