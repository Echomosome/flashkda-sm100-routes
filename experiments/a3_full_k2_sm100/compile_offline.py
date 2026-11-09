#!/usr/bin/env python3
"""无 GPU 的 CUDA/CuTe 编译核验；不把 COMPILE_PASS 写成实机正确性 PASS。

依赖已经准备好的 CUDA Toolkit 与 pinned CUTLASS，入口不会安装或下载依赖。
完整 fwd_launch.cu 编译包含所有状态/定长/变长 CUDA 模板；PyTorch binding 待实机。
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from generate import generate,LAYOUT_CHECK,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--cuda-root',type=Path,required=True)
    p.add_argument('--cutlass',type=Path,required=True)
    p.add_argument('--cuobjdump',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    a.cuda_root=a.cuda_root.resolve();a.cutlass=a.cutlass.resolve();a.cuobjdump=a.cuobjdump.resolve()
    tmp=a.out/'tmp';tmp.mkdir()
    # cuobjdump 的反汇编后端需可见；CUDA Toolkit 常将它放在同一 bin/lib 下。
    env=dict(os.environ,TMPDIR=str(tmp),
             PATH=str(a.cuobjdump.parent)+os.pathsep+os.environ.get('PATH',''),
             LD_LIBRARY_PATH=str(a.cuobjdump.parent.parent/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
    commands=[];report={}
    def command(argv,path,cwd=None):
        with path.open('w') as f:
            result=subprocess.run([str(x) for x in argv],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=cwd,timeout=1200)
        row=dict(argv=[str(x) for x in argv],exit=result.returncode,log=str(path),sha256=sha(path))
        commands.append(row)
        (a.out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        print(json.dumps(dict(event='COMMAND_EXIT',file=path.name,exit=result.returncode)),flush=True)
        if result.returncode: raise RuntimeError(f'compile or host-layout check failed; read {path}')
        return path.read_text()
    nvcc=a.cuda_root/'bin/nvcc'
    command([nvcc,'--version'],a.out/'nvcc_version.log')
    command([a.cuobjdump,'--version'],a.out/'cuobjdump_version.log')
    basic=[nvcc,'-std=c++17','-O3','-gencode=arch=compute_103a,code=sm_103a',
           '--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math',
           '--ptxas-options=-v,--register-usage-level=10,--warn-on-spills','-lineinfo','--threads','2',
           '-I'+str(a.cutlass/'include'),'-I'+str(a.cutlass/'tools/util/include'),
           '-I'+str(a.cutlass/'examples/common')]
    for route in ['A','A_control']:
        folder=a.out/route;folder.mkdir();src=generate(a.work,folder,route)
        obj=folder/'fwd_launch.o'
        log=command(basic+['-I'+str(src),'-c',src/'smxx/fwd_launch.cu','-o',obj],folder/'nvcc.log')
        resources=[];current=None;frame=None
        for line in log.splitlines():
            m=re.search(r"Compiling entry function '([^']+)'",line)
            if m: current=m.group(1)
            m=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',line)
            if m: frame=list(map(int,m.groups()))
            m=re.search(r'Used (\d+) registers',line)
            if m and current and 'recurrence' in current:
                resources.append(dict(symbol=current,registers=int(m.group(1)),stack=frame[0],
                                      spill_store_bytes=frame[1],spill_load_bytes=frame[2]))
        # 本地 cuobjdump 直接 -sass 曾发生段错误；原结果保留。
        # 提取实际 cubin 后直接 nvdisasm 成功；没有改变输入二进制或忽略机器码门槛。
        extracts=folder/'elf_extract';extracts.mkdir()
        command([a.cuobjdump,'--extract-elf','all',obj],extracts/'extract.log',cwd=extracts)
        cubins=list(extracts.glob('*.cubin'))
        if len(cubins)!=1: raise RuntimeError('expected one sm_103a cubin')
        sass=command([a.cuobjdump.with_name('nvdisasm'),'-c',cubins[0]],folder/'nvdisasm.txt')
        counts={};current=None
        for line in sass.splitlines():
            m=re.search(r'\.section\s+\.text\.(\S+)',line)
            if m: current=m.group(1);counts.setdefault(current,{})
            m=re.search(r'/\*[0-9a-fA-F]+\*/\s*(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b',line)
            if m and current and 'MMA' in m.group(1):
                op=m.group(1);counts[current][op]=counts[current].get(op,0)+1
        recurrence={n:v for n,v in counts.items() if 'recurrence' in n}
        if len(resources)!=14 or len(recurrence)!=14: raise RuntimeError('expected all 14 K2 variants')
        if route=='A':
            for ops in recurrence.values():
                if not any(op.startswith('UTCHMMA') for op in ops) or any(op.startswith('HMMA') for op in ops):
                    raise RuntimeError('A3 must have UTCHMMA and no HMMA in K2')
        else:
            for ops in recurrence.values():
                if not any(op.startswith('HMMA') for op in ops) or any(op.startswith('UTCHMMA') for op in ops):
                    raise RuntimeError('HMMA control SASS family mismatch')
        report[route]=dict(resources=resources,sass_counts=recurrence,object_sha256=sha(obj),
            object_bytes=obj.stat().st_size,CUDA_compile='PASS',GPU_execution=False)
        (folder/'compile_review.json').write_text(json.dumps(report[route],indent=2)+'\n')
        print(json.dumps(dict(event='COMPILE_REVIEW',route=route,kernels=len(recurrence),
            registers=sorted(set(x['registers'] for x in resources)),
            spill_frames=sorted(set(x['stack'] for x in resources)),
            variants=list({json.dumps(x,sort_keys=True) for x in recurrence.values()}),GPU_execution=False)),flush=True)
    # 实际 pinned CuTe 的 host 检查，链接 cudart 仅用于 CUDA 静态注册，不调用 GPU API。
    host=a.out/'layout_probe.cu'
    host.write_text('#include "fwd_kernel2.cuh"\n#include <iostream>\n'+LAYOUT_CHECK+'''
int main() {
    auto e=c1_transpose_layout_errors();
    std::cout<<"CUTE_HOST_CHECK errors="<<e<<" dynamic_smem="<<c1_transpose_smem_bytes()
             <<" GPU_execution=false\\n";
    return e?1:0;
}
''')
    binary=a.out/'layout_probe'
    cmd=basic+['-I'+str(a.out/'A/csrc/smxx'),'-L'+str(a.cuda_root/'lib'),
               '--cudart=static','--cudadevrt=static',host,'-o',binary]
    command(cmd,a.out/'layout_probe_build.log')
    host_log=command([binary],a.out/'layout_probe.log')
    print(host_log,end='',flush=True)
    report['host_layout']=host_log.strip()
    report['evidence']='OFFLINE_NVCC_CuTe_SASS; no_GPU; PyTorch_binding_not_built'
    report['full_correctness']='CUDA_PENDING';report['performance']='NOT_MEASURED'
    (a.out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__': raise SystemExit(main())
