#!/usr/bin/env python3
"""无 GPU 的 B2 交叉编译：显式指定 sm_103，不加载驱动、不执行 kernel。

开发环境需已有 Triton；本脚本不会安装包。它只能验证编译并保留 PTX/cubin，
不能生成 correctness PASS，也不能解锁 benchmark。远端仍须重新使用其 Triton 对拍。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import rescale_fused as model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--warps',type=int,default=8)
    p.add_argument('--only',choices=['all','recurrence'],default='all')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    records=[]
    for c in [16,32,64]:
        for name in ['normalize_prefix','relative_coefficients','recurrence']:
            if args.only!='all' and name!=args.only: continue
            fn=getattr(model,name)
            constants={'H':96,'C':c}
            if name=='recurrence': constants.update(HAS_INITIAL=True,HAS_FINAL=True)
            signature={}
            for key in fn.arg_names:
                if key in constants: signature[key]='constexpr'
                elif key in ['Qraw','Kraw','Graw','Braw','V','Initial','Out','Final']:
                    signature[key]='*bf16'
                elif key in ['Starts','Counts','SeqTiles']: signature[key]='*i32'
                else: signature[key]='*fp32'
            opts={'num_warps':args.warps if name=='recurrence' else (8 if c==64 else 4),
                  'num_stages':1}
            compiled=triton.compile(ASTSource(fn,signature,constexprs=constants),
                                    target=GPUTarget('cuda',103,32),options=opts)
            dest=args.out/f'C{c}_{name}';dest.mkdir()
            hashes={}
            for kind in ['ttir','ttgir','llir','ptx','cubin']:
                content=compiled.asm[kind]
                raw=content if isinstance(content,bytes) else content.encode()
                (dest/('kernel.'+kind)).write_bytes(raw)
                hashes[kind]=hashlib.sha256(raw).hexdigest()
            # ptxas 只汇编指令；-v 输出寄存器和 spill。可在完全没有 GPU 时运行。
            tool=Path(os.environ.get('TRITON_PTXAS_PATH',
                      str(Path(triton.__file__).parent/'backends/nvidia/bin/ptxas')))
            result=subprocess.run([str(tool),'-v','-arch=sm_103a',str(dest/'kernel.ptx'),
                                   '-o',str(dest/'resource_check.cubin')],text=True,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (dest/'ptxas.log').write_text(result.stdout);print(result.stdout,end='',flush=True)
            result.check_returncode()
            if compiled.metadata.shared>232448 or getattr(compiled.metadata,'tmem_size',0)>512:
                raise RuntimeError('compiled resource request exceeds B300 per-CTA limit')
            if name=='recurrence' and ('mma.' not in compiled.asm['ptx'] or 'st.global' not in compiled.asm['ptx']):
                raise RuntimeError('recurrence has no actual MMA/store body; do not ship a trap-only kernel')
            row=dict(evidence='OFFLINE_CROSS_COMPILE_NO_GPU',triton=triton.__version__,
                     C=c,kernel=name,options=opts,shared=compiled.metadata.shared,
                     tmem_columns=getattr(compiled.metadata,'tmem_size',0),
                     hashes=hashes,status='COMPILE_PASS',correctness='NOT_RUN',performance='NOT_RUN')
            records.append(row);print(json.dumps(row),flush=True)
            (args.out/'compile.json').write_text(json.dumps(records,indent=2)+'\n')
    return 0


if __name__=='__main__': raise SystemExit(main())
