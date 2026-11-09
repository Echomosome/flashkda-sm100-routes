"""Command line entry point.

Examples
--------
    python -m flashkda_routes.cli offline
    python -m flashkda_routes.cli correctness --routes A B C --suite full
    python -m flashkda_routes.cli benchmark --routes C --validated runs/build/<id>

GPU work runs in a separate worker process per route, so a CUDA fault in one
candidate cannot contaminate the others.  No command installs packages,
overwrites the baseline, or appends to README.md.
"""

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from .config import DEFAULT_WORK, REPO_ROOT
from .emit import emit
from .manifest import write_runner_manifest


def _new_build(work, stage):
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    build = Path(work) / "build" / f"{run_id}_flashkda_routes_{stage}"
    build.mkdir(parents=True, exist_ok=False)
    snapshot = build / "runner_snapshot"
    snapshot.mkdir()
    for src in sorted((REPO_ROOT / "flashkda_routes").rglob("*")):
        if src.is_file() and src.suffix in (".py", ".cuh"):
            dst = snapshot / src.relative_to(REPO_ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    manifest = write_runner_manifest(build)
    (build / "runner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return run_id, build


def _gpu_snapshot(path):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=30)
    except FileNotFoundError:
        path.write_text("nvidia-smi not found\n", encoding="utf-8")
        return
    path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        emit("GPU_SNAPSHOT", file=str(path), exit=result.returncode)


def _worker(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python")
    props = torch.cuda.get_device_properties(0)
    emit("DEVICE", evidence="REAL_CUDA_WORKER", name=props.name,
         capability=[props.major, props.minor], SMs=props.multi_processor_count,
         torch=torch.__version__, cuda=torch.version.cuda,
         route=args.route, stage=args.stage, suite=args.suite)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.stage == "correctness":
        from .correctness import run
        with torch.inference_mode():
            run(args.work, args.build, args.route, args.suite,
                arch=args.arch, baseline_module=args.baseline_module,
                strict_evidence=args.strict_evidence)
    else:
        from .benchmark import run
        with torch.inference_mode():
            run(args.work, args.build, args.validated, args.route,
                baseline_module=args.baseline_module, rounds=args.rounds,
                warmup=args.warmup, strict_evidence=args.strict_evidence)
    return 0


def _spawn_routes(args, build, stage):
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.prefix) / "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.setdefault("MAX_JOBS", "2")
    codes = {}
    for route in dict.fromkeys(args.routes):
        before = build / f"gpu_before_{route}.csv"
        _gpu_snapshot(before)
        cmd = [sys.executable, "-u", "-m", "flashkda_routes.cli", "worker",
               "--stage", stage, "--route", route, "--suite", args.suite,
               "--work", str(args.work.resolve()), "--build", str(build.resolve()),
               "--arch", args.arch, "--baseline-module", args.baseline_module]
        if stage == "benchmark":
            cmd += ["--validated", str(args.validated.resolve()),
                    "--rounds", str(args.rounds), "--warmup", str(args.warmup)]
        if args.strict_evidence:
            cmd.append("--strict-evidence")
        emit("WORKER_COMMAND", route=route, argv=cmd)
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)

        def timeout(p=proc):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(args.timeout_seconds, timeout)
        timer.start()
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
            codes[route] = proc.wait()
        finally:
            timer.cancel()
        _gpu_snapshot(build / f"gpu_after_{route}.csv")
        emit("WORKER_EXIT", route=route, exit=codes[route])
        verdict = build / f"verdict_{route}.json"
        if codes[route] and not verdict.exists():
            verdict.write_text(json.dumps(
                dict(route=route, status="WORKER_FAILED", exit=codes[route]), indent=2) + "\n", encoding="utf-8")
    return codes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work", type=Path, default=DEFAULT_WORK,
                        help="run directory whose parent contains FlashKDA/ and fla_kda_ref/")
    common.add_argument("--timeout-seconds", type=int, default=3600)
    common.add_argument("--arch", default="10.3a", help="CUDA arch, e.g. 10.3a or 9.0a")
    common.add_argument("--baseline-module", default="flash_kda_C")
    common.add_argument("--strict-evidence", action="store_true",
                        help="require the recorded B300 baseline binary hash")

    sub.add_parser("offline", parents=[common], help="CPU-only source and algebra checks")
    p_cor = sub.add_parser("correctness", parents=[common], help="full-KDA correctness gate")
    p_cor.add_argument("--routes", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    p_cor.add_argument("--suite", choices=["short", "full"], default="short")
    p_ben = sub.add_parser("benchmark", parents=[common], help="complete-forward CUDA Graph timing")
    p_ben.add_argument("--routes", nargs="+", choices=["A", "B", "C"], required=True)
    p_ben.add_argument("--validated", type=Path, required=True,
                       help="build directory with a PASS/full verdict")
    p_ben.add_argument("--rounds", type=int, default=12)
    p_ben.add_argument("--warmup", type=int, default=5)
    p_ben.set_defaults(suite="full")

    p_work = sub.add_parser("worker")
    p_work.add_argument("--stage", choices=["correctness", "benchmark"], required=True)
    p_work.add_argument("--route", choices=["A", "B", "C"], required=True)
    p_work.add_argument("--suite", choices=["short", "full"], default="short")
    p_work.add_argument("--work", type=Path, required=True)
    p_work.add_argument("--build", type=Path, required=True)
    p_work.add_argument("--validated", type=Path)
    p_work.add_argument("--arch", default="10.3a")
    p_work.add_argument("--baseline-module", default="flash_kda_C")
    p_work.add_argument("--rounds", type=int, default=12)
    p_work.add_argument("--warmup", type=int, default=5)
    p_work.add_argument("--strict-evidence", action="store_true")

    args = parser.parse_args(argv)
    args.work = args.work.resolve()

    if args.command == "worker":
        return _worker(args)

    if args.command == "offline":
        from .offline import run as offline_run
        run_id, build = _new_build(args.work, "offline")
        emit("PURPOSE", task="flashkda_routes_offline", mode="offline", stage="offline",
             run_id=run_id, python=sys.executable, build=str(build))
        try:
            offline_run(args.work, build)
            status = 0
        except Exception:
            traceback.print_exc()
            status = 1
        emit("RUN_COMPLETE", exit=status, evidence="OFFLINE_CPU", performance="NOT_MEASURED")
        return status

    if args.command == "benchmark" and args.validated is None:
        parser.error("benchmark requires --validated")

    stage = args.command
    run_id, build = _new_build(args.work, stage)
    emit("PURPOSE", task=f"flashkda_routes_{stage}", mode="cuda", stage=stage,
         suite=args.suite, run_id=run_id, python=sys.executable, build=str(build),
         slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    codes = _spawn_routes(args, build, stage)
    status = 1 if any(codes.values()) else 0
    receipt = dict(mode="cuda", stage=stage, suite=args.suite, exit=status,
                   build=str(build), route_exits=codes)
    (build / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    emit("RECEIPT", **receipt)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
