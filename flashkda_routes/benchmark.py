"""Complete-forward timing.  Requires a PASS/full verdict from the same build.

One CUDA Graph holds one complete forward.  Twelve rounds alternate the order
of baseline and candidate.  Outputs are poisoned and replayed after timing to
prove the graph covers every output/state element.  ``B`` uses FP32 internal
state and a different precision path; the table keeps that distinction.
"""

import json
import statistics
from pathlib import Path

import torch

from .build import load_module
from .cases import Native, Rescaled, bit_mismatches, cases, fixture
from .config import RECORDED_BASELINE_SHA, sha256
from .emit import emit
from .manifest import verify_runner_manifest


def run(work, build, validated, route, baseline_module="flash_kda_C",
        rounds=12, warmup=5, strict_evidence=False):
    import importlib
    work, build, validated = Path(work), Path(build), Path(validated)
    verdict = json.loads((validated / f"verdict_{route}.json").read_text(encoding="utf-8"))
    if verdict.get("status") != "PASS" or verdict.get("suite") != "full":
        raise RuntimeError("benchmark requires a PASS/full verdict for this route")
    baseline = importlib.import_module(baseline_module)
    baseline_sha = sha256(baseline.__file__)
    if strict_evidence and baseline_sha != RECORDED_BASELINE_SHA:
        raise RuntimeError("baseline binary changed relative to the recorded run")
    verify_runner_manifest(validated)

    emit("BENCH_SCOPE", route=route, validated=str(validated), clocks="unmodified",
         cache_flush=False, graph_forward_calls=1, rounds=rounds, warmup=warmup,
         mode="complete_forward_GPU_Graph",
         included="normalization_gate_beta_prepare_recurrence_state_output",
         B_precision="FP32_internal_and_triangular_solver; BF16_IO")

    implementations = {}
    if route == "B":
        implementations = {f"B{c}": c for c in [16, 32, 64]}
    else:
        for name in (["A_control", "A"] if route == "A" else ["C"]):
            identity = json.loads((validated / name / "binary.json").read_text(encoding="utf-8"))
            if sha256(identity["path"]) != identity["sha256"]:
                raise RuntimeError("candidate binary changed")
            implementations[name] = load_module(identity["name"], identity["path"])

    records = []

    def record(tag, **row):
        records.append(dict(tag=tag, **row))
        (build / f"bench_{route}.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        emit(tag, **row)

    selected = {"fixed_h96", "varlen6_h96", "varlen8_h96", "fixed_h12"}
    for index, spec in enumerate(cases("full")):
        if spec["name"] not in selected:
            continue
        data, initial, dtype, want_final, cu = fixture(spec, 2026091100 + index)
        frozen = None if initial is None else initial.clone()
        for name, module in implementations.items():
            candidate = (Rescaled(module, data, initial, dtype, want_final, cu, spec)
                         if route == "B" else Native(module, data, initial, dtype, want_final, cu))
            runners = {"baseline": Native(baseline, data, initial, dtype, want_final, cu),
                       name: candidate}
            graphs, outputs, gold = {}, {}, {}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            for impl, runner in runners.items():
                with torch.cuda.stream(stream):
                    for _ in range(warmup):
                        eager = runner.run()
                    gold[impl] = tuple(x.clone() for x in eager)
                torch.cuda.current_stream().wait_stream(stream)
                graphs[impl] = torch.cuda.CUDAGraph()
                # If a library call cannot be captured, fail rather than switch
                # to a different timing method silently.
                with torch.cuda.graph(graphs[impl], stream=stream):
                    outputs[impl] = runner.run()
                for _ in range(3):
                    graphs[impl].replay()
                torch.cuda.synchronize()
                for i, label in enumerate(["output", "state"]):
                    if bit_mismatches(outputs[impl][i], gold[impl][i]):
                        raise RuntimeError(f"graph/eager mismatch: {impl}/{label}")
            samples = {key: [] for key in runners}
            keys = list(runners)
            for round_id in range(rounds):
                order = keys if round_id % 2 == 0 else keys[::-1]
                for slot, impl in enumerate(order):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[impl].replay()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    samples[impl].append(ms)
                    record("RAW_TIME", case=spec["name"], impl=impl, round=round_id,
                           slot=slot, ms_per_forward=ms)
            for impl in runners:
                # Poison every output/state element, replay, and require exact
                # recovery.  This proves the graph really writes all of them.
                for x in outputs[impl]:
                    x.fill_(float("nan"))
                graphs[impl].replay()
                torch.cuda.synchronize()
                for i, label in enumerate(["output", "state"]):
                    if not torch.isfinite(outputs[impl][i]).all() or \
                            bit_mismatches(outputs[impl][i], gold[impl][i]):
                        raise RuntimeError(f"post-timing poison check failed: {impl}/{label}")
                    record("POSTCHECK", case=spec["name"], impl=impl, tensor=label, status="PASS")
            if initial is not None and bit_mismatches(initial, frozen):
                raise RuntimeError("initial state mutated during benchmark")
            tb = statistics.median(samples["baseline"])
            tc = statistics.median(samples[name])
            record("COMPARISON", case=spec["name"], impl=name,
                   baseline_median_ms=tb, candidate_median_ms=tc, speedup=tb / tc,
                   raw_samples=samples, status="ACCEPTED_AFTER_POSTCHECK",
                   precision="FP32_internal_solver" if route == "B"
                   else "baseline_BF16_state_boundaries")
            del graphs, outputs, gold, runners, candidate
        del data, initial, frozen
    emit("BENCHMARK_COMPLETE", route=route, status="PASS",
         scope="these_implementations_and_four_shapes_only")
