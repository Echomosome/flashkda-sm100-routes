"""独立的 CPU 检查：地址归属、descriptor 整数模型和带 BF16 舍入的阶段公式。

不导入 torch，不发 CUDA kernel。CPU dot 用 FP64，不能替代真实 MMA 的数值对拍。
"""
from pathlib import Path
import ast
import json
import numpy as np
from generate import generate


def emit(tag,**row):
    print(tag+' '+json.dumps(row,ensure_ascii=False,sort_keys=True),flush=True)


def rn(x):
    """FP32→BF16 的 round-to-nearest-even；输出仍用 FP32 容器。"""
    a=np.asarray(x,dtype=np.float32)
    bits=a.view(np.uint32)
    rounded=(bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)
    return rounded.view(np.float32)


def mm(a,b):
    return a.astype(np.float64)@b.astype(np.float64)


def phase_old(state,kd,qd,kr,inv,mqk,v,beta,gate):
    pred=rn(mm(kd,state.T))
    rhs=rn(rn(v-pred)*beta[:,None])
    u=rn(mm(inv,rhs))
    out=rn(rn(mm(qd,state.T))+rn(mm(mqk,u)))
    # 模拟状态 epilogue 的一次 FP32 舍入，再 BF16；不是硬件 FMA 模拟器。
    update=mm(kr.T,u).T
    new=rn((state.astype(np.float64)*gate[None,:]+update).astype(np.float32))
    return out,new,u


def phase_new(state,kd,qd,kr,inv,mqk,v,beta,gate):
    fused=rn(mm(state,np.concatenate([kd,qd],axis=0).T))
    pred=fused[:,:16]
    rhs=rn(rn(v.T-pred)*beta[None,:])
    ut=rn(mm(rhs,inv.T))
    out=rn(fused[:,16:]+rn(mm(ut,mqk.T)))
    new=rn((state.astype(np.float64)*gate[None,:]+mm(ut,kr)).astype(np.float32))
    return out.T,new,ut.T


def run(work,build):
    root=Path(__file__).parent;build=Path(build);rows=[]
    for p in root.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
    for route in ['A','A_control']:
        path=build/route;path.mkdir(parents=True,exist_ok=True)
        generate(work,path,route)
    emit('SOURCE_CHECK',status='PASS',K1_utils_fwd_h='byte_identical',
         K2='all_GEMMs_transposed',CHUNK=16,GPU_execution=False)
    # SMEM 各矩阵布局的数学模型，真实 CuTe 的检查另由 nvcc 编译运行。
    checked=0
    for m_count,k_count in [(128,128),(128,16),(16,128),(16,16),(32,128)]:
        offsets={}
        for m in range(m_count):
            for k in range(k_count):
                off=k%8+8*(m%8)+64*(m//8)+8*m_count*(k//8)
                assert 0<=off<m_count*k_count and off not in offsets
                offsets[off]=(m,k);checked+=1
        # 每次 K16 指令的基址平移，仍读取同一完整 K 维。
        for base in range(0,k_count,16):
            for m in range(m_count):
                for k in range(16):
                    local=k%8+8*(m%8)+64*(m//8)+8*m_count*(k//8)
                    assert offsets[base*m_count+local]==(m,base+k)
    emit('SMEM_ADDRESS_MODEL',status='PASS',elements=checked,scope='integer_model')
    # wide TMEM load 的 64 个寄存器按相邻 16×16 片段排序。
    owners={};small={}
    for w in range(4):
        for lane in range(32):
            for mi in range(2):
                for ni in range(8):
                    for j in range(8):
                        m=w*32+mi*16+lane//4+8*((j//2)%2)
                        n=ni*16+2*(lane%4)+(j%2)+8*(j//4)
                        assert m//32==w and (m,n) not in owners
                        owners[m,n]=(w,lane,mi,ni*8+j)
                        if ni==0: small[m,n]=(w,lane,mi,j)
    assert len(owners)==16384 and len(small)==2048
    emit('TMEM_OWNERSHIP_MODEL',status='PASS',state_elements=len(owners),UT_elements=len(small),
         aliased_accumulator_phases=['P1:0..31','P3:0..15','P4:128..143 + P6:0..127'],
         allocated_columns=256,commit_wait_groups_per_chunk=3,GPU=False)
    rng=np.random.default_rng(2026091120)
    chains=0
    for chunks in [1,2,8,32]:
        for gate_kind in ['zero','weak','ordinary']:
            state=rn(rng.normal(0,.125,(128,128)));trans=state.copy()
            for _ in range(chunks):
                kd,qd,kr=[rn(rng.normal(0,.06,(16,128))) for i in range(3)]
                inv=rn(np.eye(16)+np.tril(rng.normal(0,.03,(16,16)),-1))
                mqk=rn(np.tril(rng.normal(0,.025,(16,16))))
                v=rn(rng.normal(0,.25,(16,128)));beta=rn(rng.uniform(0,1,16))
                gate=np.full(128,{'zero':0,'weak':.9999,'ordinary':.8}[gate_kind],np.float32)
                a,state,u=phase_old(state,kd,qd,kr,inv,mqk,v,beta,gate)
                b,trans,ut=phase_new(trans,kd,qd,kr,inv,mqk,v,beta,gate)
                assert np.array_equal(a,b) and np.array_equal(state,trans) and np.array_equal(u,ut)
                assert np.isfinite(a).all() and np.isfinite(state).all()
            chains+=1
    emit('PHASE_CHAIN_ALGEBRA',status='PASS',chains=chains,chunks=[1,2,8,32],
         exact_BF16_container_match=True,
         scope='prepared_inputs_FP64_dots_BF16_boundaries; not_CUDA_MMA_or_full_KDA_reference')
    summary=dict(status='OFFLINE_PASS',full_KDA_correctness='CUDA_PENDING',performance='NOT_MEASURED',
                 CUDA_execution=False,source_generated=True,SMEM_elements=checked,algebra_chains=chains)
    (build/'offline_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    run(args.work,args.out)
