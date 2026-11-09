"""Shared inputs, reference metrics and runner wrappers for all routes."""

import torch

from .routes.b_rescale import raw_forward


def metrics(actual, reference):
    """NRMSE is a ratio, not a percentage; also report max error and non-finites."""
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(finite=False, nrmse=None, max_abs=None,
                    nonfinite=int((~torch.isfinite(a)).sum()),
                    reference_nonfinite=int((~torch.isfinite(b)).sum()))
    delta = a - b
    nr = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(b).clamp_min(1e-30))
    return dict(finite=True, nrmse=nr, max_abs=float(delta.abs().max()), nonfinite=0,
                reference_nonfinite=0)


def bit_mismatches(a, b):
    """Number of bitwise differences between two tensors of the same dtype."""
    if a.dtype == torch.bfloat16:
        dtype = torch.int16
    elif a.dtype == torch.float64:
        dtype = torch.int64
    else:
        dtype = torch.int32
    return int((a.view(dtype) != b.view(dtype)).sum().item())


def cases(suite):
    """Short matrix for localisation; ``full`` adds the four long shapes."""
    result = []
    modes = ["none", "in_bf16", "out_bf16", "both_bf16", "in_f32", "out_f32", "both_f32"]
    for variable in (False, True):
        for mode in modes:
            result.append(dict(name=("varlen_" if variable else "fixed_") + mode, time=65,
                               heads=4, batch=1, lengths=[17, 33, 65] if variable else None,
                               mode=mode, gate="random"))
    result.append(dict(name="batch2", time=65, heads=4, batch=2, lengths=None,
                       mode="both_f32", gate="random"))
    for gate in ["strong5", "strong2_5", "zero", "weak", "coherent", "zero_qk"]:
        result.append(dict(name=gate, time=513 if gate == "weak" else 129, heads=4, batch=1,
                           lengths=None, mode="both_bf16", gate=gate))
    for t in [1, 15, 16, 17, 31, 32, 33, 63, 64, 65]:
        result.append(dict(name=f"tail_{t}", time=t, heads=1, batch=1, lengths=None,
                           mode="both_bf16", gate="random"))
    for h in [12, 96]:
        result.append(dict(name=f"grid_h{h}", time=65, heads=h, batch=1, lengths=None,
                           mode="both_bf16", gate="random"))
    if suite == "full":
        for name, h, ls in [("fixed_h96", 96, None),
                            ("varlen6_h96", 96, [1300, 547, 2048, 963, 271, 3063]),
                            ("varlen8_h96", 96, [1024] * 8),
                            ("fixed_h12", 12, None),
                            ("weak_long", 4, None)]:
            result.append(dict(name=name, time=8192, heads=h, batch=1, lengths=ls,
                               mode="both_bf16", gate="weak" if name == "weak_long" else "random"))
    return result


def fixture(spec, seed):
    """Each case gets its own CPU seed, independent of route execution order."""
    t = sum(spec["lengths"]) if spec["lengths"] else spec["time"]
    shape = (spec["batch"], t, spec["heads"], 128)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    normal = lambda s: torch.randn(s, generator=rng, dtype=torch.float32)
    q, k = normal(shape), normal(shape)
    v = normal(shape) * torch.linspace(.25, 1.25, 128) / 4
    g = normal(shape) * .5 - 2
    b = normal(shape[:-1]) * .5
    alog = torch.linspace(-.15, .15, spec["heads"])
    bias = torch.linspace(-.2, .2, spec["heads"] * 128).reshape(spec["heads"], 128)
    kind = spec["gate"]
    if kind in ("strong5", "strong2_5", "zero", "weak", "coherent"):
        alog.zero_()
        bias.zero_()
        g.fill_({"strong5": 20., "strong2_5": 0., "zero": -80., "weak": -16., "coherent": -16.}[kind])
        b.fill_(-4. if kind == "weak" else 0.)
    if kind == "coherent":
        q.fill_(1)
        k.fill_(1)
    if kind == "zero_qk":
        q.zero_()
        k.zero_()
    data = [x.to(torch.bfloat16).cuda() for x in (q, k, v, g, b)] + [alog.cuda(), bias.cuda()]
    mode = spec["mode"]
    dtype = torch.float32 if mode.endswith("f32") else torch.bfloat16
    nin = len(spec["lengths"]) if spec["lengths"] else spec["batch"]
    initial = (normal((nin, spec["heads"], 128, 128)) / 8).to(dtype).cuda() \
        if mode.startswith(("in", "both")) else None
    want_final = mode.startswith(("out", "both"))
    cu = None
    if spec["lengths"]:
        prefix = [0]
        for x in spec["lengths"]:
            prefix.append(prefix[-1] + x)
        cu = torch.tensor(prefix, dtype=torch.int64, device="cuda")
    return tuple(data), initial, dtype, want_final, cu


class Native:
    """Pre-allocated output/workspace; ``run`` contains the complete forward."""

    def __init__(self, module, data, initial, dtype, want_final, cu):
        self.module, self.data, self.initial, self.cu = module, data, initial, cu
        b, t, h, d = data[0].shape
        n = cu.numel() - 1 if cu is not None else b
        self.out = torch.full_like(data[2], float("nan"))
        self.final = torch.full((n, h, d, d), float("nan"), device="cuda", dtype=dtype) \
            if want_final else None
        self.ws = torch.empty(module.get_workspace_size(b * t, h, n),
                              device="cuda", dtype=torch.uint8)

    def run(self):
        q, k, v, g, b, a, bias = self.data
        self.module.fwd(q, k, v, g, b, 128 ** -.5, self.out, self.ws, a, bias, -5.,
                        initial_state=self.initial, final_state=self.final, cu_seqlens=self.cu)
        return self.out, self.final


class Rescaled:
    """Route B wrapper: FP32 internal state and unit-lower-triangular solve."""

    def __init__(self, chunk, data, initial, dtype, want_final, cu, spec):
        self.chunk, self.data, self.initial, self.dtype = chunk, data, initial, dtype
        self.want_final, self.lengths = want_final, spec["lengths"]

    def run(self):
        return raw_forward(self.data, self.initial, self.dtype, self.lengths,
                           self.chunk, self.want_final)
