"""Full-KDA correctness gate.  Thresholds live in code and are never relaxed
after seeing a result.

Every case compares the complete output and the returned final state against
the pinned FLA naive recurrence, and checks that the read-only initial state
is unchanged.  A failure keeps the input/output arrays for inspection.
"""

import importlib
import json
from pathlib import Path

import torch

from .build import build_native
from .cases import (
    Native, Rescaled, bit_mismatches, cases, fixture, metrics,
)
from .config import RECORDED_BASELINE_SHA, sha256
from .emit import emit
from .reference import naive_reference


def _load_baseline(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"baseline extension {name!r} is not importable; build/install the "
            "pinned FlashKDA first (see README)") from exc


def run(work, build, route, suite, arch="10.3a", baseline_module="flash_kda_C",
        strict_evidence=False):
    work, build = Path(work), Path(build)
    baseline = _load_baseline(baseline_module)
    baseline_sha = sha256(baseline.__file__)
    if strict_evidence and baseline_sha != RECORDED_BASELINE_SHA:
        raise RuntimeError(
            f"installed baseline {baseline.__file__} has {baseline_sha}, expected "
            f"{RECORDED_BASELINE_SHA} (--strict-evidence)")
    emit("BASELINE_BINARY", file=baseline.__file__, sha256=baseline_sha,
         strict_match=baseline_sha == RECORDED_BASELINE_SHA)

    impls = {}
    if route == "A":
        # Build the control first; both keep independent patches, binaries and SASS.
        for name in ["A_control", "A"]:
            impls[name] = build_native(work, build / name, name, arch=arch)
    elif route == "C":
        impls["C"] = build_native(work, build / "C", "C", arch=arch)
    elif route == "B":
        impls = {f"B{c}": c for c in [16, 32, 64]}  # C16 is the solver/precision control
        emit("B_IMPLEMENTATION", chunks=[16, 32, 64], state_internal="FP32",
             solve="FP32_unit_lower_triangular", rescale="pairwise_log_prefix_rebase",
             backend="PyTorch_CUDA_composite", fused=False, tcgen05_claim=False)
    else:
        raise ValueError(route)

    verdict_path = build / f"verdict_{route}.json"
    records = []
    failures = 0
    case_count = 0
    # Engineering gates, not model-quality thresholds.
    emit("ACCEPTANCE_POLICY", route=route,
         B="FLA NRMSE<=0.005 and max_abs<=0.01; finite; FP32 state/solve",
         A="baseline NRMSE<=0.005 and max_abs<=0.01; finite; plus FLA envelope",
         C="bitwise baseline including final_state; plus FLA envelope",
         FLA_envelope="NRMSE<=min(0.05,max(0.015,1.10*baseline_NRMSE+0.001)); "
                      "max_abs<=max(0.02,1.10*baseline_max_abs+0.002)")

    def record(tag, **row):
        records.append(dict(tag=tag, **row))
        (build / f"checks_{route}.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        emit(tag, **row)

    try:
        for index, spec in enumerate(cases(suite)):
            case_count += 1
            seed = 2026091100 + index
            emit("CASE_START", route=route, seed=seed, **spec)
            data, initial, dtype, want_final, cu = fixture(spec, seed)
            frozen = None if initial is None else initial.clone()
            ref = naive_reference(work, data, initial, spec)
            native_base = Native(baseline, data, initial, dtype, want_final, cu)
            base = native_base.run()
            torch.cuda.synchronize()
            if initial is not None and bit_mismatches(initial, frozen):
                raise RuntimeError("baseline changed read-only initial state")
            base_metrics = {}
            for i, label in enumerate(["output", "state"]):
                if i == 1 and not want_final:
                    continue
                base_metrics[label] = metrics(base[i], ref[i])
                record("BASELINE_VS_FLA", case=spec["name"], tensor=label, **base_metrics[label])
                if not base_metrics[label]["finite"]:
                    raise RuntimeError("nonfinite baseline/reference; cannot perform relative acceptance")
            for name, module in impls.items():
                runner = (Rescaled(module, data, initial, dtype, want_final, cu, spec)
                          if name.startswith("B") else Native(module, data, initial, dtype, want_final, cu))
                actual = runner.run()
                torch.cuda.synchronize()
                bad = False
                for i, label in enumerate(["output", "state"]):
                    if i == 1 and not want_final:
                        continue
                    fm = metrics(actual[i], ref[i])
                    bm = metrics(actual[i], base[i])
                    bref = base_metrics[label]
                    bits = bit_mismatches(actual[i], base[i])
                    envelope = min(.05, max(.015, 1.10 * bref["nrmse"] + .001))
                    maxabs = max(.02, 1.10 * bref["max_abs"] + .002)
                    if name.startswith("B"):
                        good = fm["finite"] and fm["nrmse"] <= .005 and fm["max_abs"] <= .01
                    else:
                        parity = (bits == 0) if name == "C" else (
                            bm["finite"] and bm["nrmse"] <= .005 and bm["max_abs"] <= .01)
                        good = parity and fm["finite"] and fm["nrmse"] <= envelope \
                            and fm["max_abs"] <= maxabs
                    bad |= not good
                    record("CANDIDATE_COMPARE", case=spec["name"], impl=name, tensor=label,
                           elements=actual[i].numel(), vs_fla=fm, vs_baseline=bm,
                           bit_mismatches=bits, status="PASS" if good else "FAIL")
                unchanged = initial is None or bit_mismatches(initial, frozen) == 0
                record("INITIAL_UNCHANGED", case=spec["name"], impl=name,
                       status="PASS" if unchanged else "FAIL")
                bad |= not unchanged
                if bad:
                    failures += 1
                    if failures <= 2:
                        torch.save(dict(
                            spec=spec, data=[x.cpu() for x in data],
                            initial=None if frozen is None else frozen.cpu(),
                            candidate=[x.cpu() if x is not None else None for x in actual],
                            baseline=[x.cpu() if x is not None else None for x in base],
                            reference=ref),
                            build / f'failure_{route}_{failures}_{name}_{spec["name"]}.pt')
                    if not unchanged:
                        raise RuntimeError("candidate changed initial state; stop contaminated comparisons")
                del actual, runner
            del data, initial, frozen, ref, base, native_base
        status = "PASS" if failures == 0 else "FAIL"
        verdict = dict(route=route, status=status, suite=suite, cases=case_count,
                       failed_comparisons=failures, baseline_sha256=baseline_sha,
                       reference="pinned_fla_kda_ref_naive_recurrent_FP32",
                       scope="only_tested_full_KDA_inputs; no_model_quality_claim",
                       performance="NOT_MEASURED")
        verdict_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        emit("ROUTE_VERDICT", **verdict)
        if failures:
            raise RuntimeError(f"{route}: numerical gate failed; timing disabled")
    except Exception as exc:
        if not verdict_path.exists():
            verdict_path.write_text(json.dumps(dict(
                route=route, status="RUNTIME_FAILED", suite=suite,
                cases_started=case_count, error=str(exc), performance="NOT_MEASURED"), indent=2) + "\n", encoding="utf-8")
        raise
