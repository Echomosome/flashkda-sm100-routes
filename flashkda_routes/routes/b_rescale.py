"""Route B: a real C=32/64 KDA with relative-prefix rebasing.

This is a PyTorch CUDA composite implementation for numerical comparison.  It
is *not* a fused kernel and makes no tcgen05 claim.  The same formulas are used
by the NumPy offline audit, so the two versions cannot drift apart silently.

The point of the rebasing is to avoid materialising ``exp(-G_j)``: for a causal
position ``i >= j`` with non-positive gates, ``G_i - G_j <= 0`` and every
``exp(G_i - G_j)`` is in [0, 1].  The non-causal entries are masked *before*
the exponential, never after an Inf has been produced.

Reading order: ``G``/decay -> ``L``/RHS -> triangular solve -> output/state in
``chunk_core``, then ``raw_forward`` for input activation and layout transposes.
"""

import numpy as np


def bf16_numpy(x):
    """Round float32 to BF16 (round-to-nearest-even); used by offline checks."""
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


class Ops:
    """The few NumPy/PyTorch differences; the formulas are shared."""

    def __init__(self, xp, sample):
        self.xp, self.sample = xp, sample
        self.is_numpy = xp is np

    def eye(self, n):
        return np.eye(n, dtype=self.sample.dtype) if self.is_numpy else self.xp.eye(
            n, dtype=self.sample.dtype, device=self.sample.device)

    def mask(self, n, strict=False):
        ids = np.arange(n) if self.is_numpy else self.xp.arange(n, device=self.sample.device)
        return ids[:, None] > ids[None, :] if strict else ids[:, None] >= ids[None, :]

    def transpose(self, a):
        return a.swapaxes(-1, -2) if self.is_numpy else a.transpose(-1, -2)

    def sum(self, a):
        return a.sum(axis=-1) if self.is_numpy else a.sum(dim=-1)

    def clone(self, a):
        return a.copy() if self.is_numpy else a.clone()

    def triangular_solve(self, a, rhs):
        if not self.is_numpy:
            # Unit lower triangular: no explicit inverse and no FP16 Neumann series.
            return self.xp.linalg.solve_triangular(a, rhs, upper=False, unitriangular=True)
        out = rhs.copy()
        for i in range(rhs.shape[-2]):
            if i:
                out[:, i] -= (a[:, i, :i, None] * out[:, :i]).sum(axis=1)
        return out


def chunk_core(q, k, v, g, beta, state, chunk, xp=np, rescale=True):
    """Activated inputs ``[H,T,D]``; state ``[H,key,value]`` in FP32/FP64.

    ``G_i = sum_{t<=i} g_t``.  For ``i >= j`` compute ``exp(G_i-G_j)`` directly,
    rebased at the contributing token ``j``.  With legal gates (<= 0) every
    factor is in [0, 1].  ``rescale=False`` is an offline negative control that
    deliberately keeps the overflowing absolute-prefix factorisation.
    """
    if chunk not in (16, 32, 64):
        raise ValueError("supported CHUNK: 16/32/64")
    op = Ops(xp, q)
    pieces = []
    state = op.clone(state)  # the initial state is read-only input
    for start in range(0, q.shape[1], chunk):
        end = min(start + chunk, q.shape[1])
        qc, kc, vc, bc = q[:, start:end], k[:, start:end], v[:, start:end], beta[:, start:end]
        gc = g[:, start:end]
        G = np.cumsum(gc, axis=1) if xp is np else xp.cumsum(gc, dim=1)
        size = end - start
        causal = op.mask(size)[None, :, :, None]
        if rescale:
            delta = G[:, :, None, :] - G[:, None, :, :]
            # Mask future positions BEFORE exp: never produce Inf and zero it later.
            delta = xp.where(causal, delta, xp.zeros_like(delta))
            decay = xp.where(causal, xp.exp(delta), xp.zeros_like(delta))
        else:
            # Negative control: demonstrates 0 * Inf; never used by the CUDA route.
            decay = xp.exp(G[:, :, None, :]) * xp.exp(-G[:, None, :, :])
            decay = xp.where(causal, decay, xp.zeros_like(decay))
        # L_ij = beta_i <k_i, exp(G_i-G_j) k_j>, i > j
        # RHS_i = beta_i [v_i - (k_i exp(G_i)) S_start]
        L = bc[:, :, None] * op.sum(kc[:, :, None, :] * decay * kc[:, None, :, :])
        L = xp.where(op.mask(size, strict=True)[None], L, xp.zeros_like(L))
        prefix_decay = xp.exp(G)
        rhs = bc[:, :, None] * (vc - (kc * prefix_decay) @ state)
        U = op.triangular_solve(op.eye(size)[None] + L, rhs)
        # Include the diagonal: the current token is written before its output.
        QK = op.sum(qc[:, :, None, :] * decay * kc[:, None, :, :])
        pieces.append((qc * prefix_decay) @ state + QK @ U)
        # Rebuild the state in real coordinates at the chunk boundary, then
        # restart the prefix from zero in the next chunk.
        # G_end - G_i <= 0, so no inverse decay is needed at the tail either.
        kr = kc * xp.exp(G[:, -1:, :] - G)
        state = state * xp.exp(G[:, -1, :, None]) + op.transpose(kr) @ U
    out = np.concatenate(pieces, axis=1) if xp is np else xp.cat(pieces, dim=1)
    return out, state


def activate_torch(data):
    """Match the raw FlashKDA input convention; no A_log=dt_bias=0 shortcut."""
    import torch
    q, k, v, raw_g, raw_beta, alog, bias = data
    q, k = q.float(), k.float()
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    # FlashKDA uses tanh.approx sigmoid; this is FP32 torch sigmoid and is
    # compared separately.
    g = -5.0 * torch.sigmoid((raw_g.float() + bias[None, None]) * alog.exp()[None, None, :, None])
    return q * 128 ** -0.5, k, v.float(), g, torch.sigmoid(raw_beta.float())


def raw_forward(data, initial, state_dtype, lengths, chunk, want_final=True):
    """Full mathematical equivalent of the raw API; BF16 output, FP32 state/solve.

    ``lengths=None`` means a fixed batch; otherwise the input is B=1 packed
    varlen and each sequence is processed independently.
    """
    import torch
    q, k, v, g, b = activate_torch(data)
    batch, time, heads, dim = q.shape
    if lengths is not None and (batch != 1 or sum(lengths) != time or min(lengths) <= 0):
        raise ValueError("invalid packed sequence lengths")
    seq_lens = [time] * batch if lengths is None else lengths
    outs, states, bos = [], [], 0
    for n, count in enumerate(seq_lens):
        bid, start = (n, 0) if lengths is None else (0, bos)
        seq = [x[bid, start:start + count].transpose(0, 1) for x in (q, k, v, g, b)]
        s0 = (torch.zeros((heads, dim, dim), device=q.device, dtype=torch.float32)
              if initial is None else initial[n].float().transpose(-1, -2))
        out, state = chunk_core(*seq, s0, chunk, xp=torch)
        outs.append(out.transpose(0, 1)[None])
        states.append(state.transpose(-1, -2)[None])
        bos += count
    output = torch.cat(outs, dim=0 if lengths is None else 1).to(torch.bfloat16)
    final = torch.cat(states, dim=0).to(state_dtype).contiguous() if want_final else None
    return output, final


def recurrent_numpy(q, k, v, g, beta, initial):
    """Independent token recurrence used to check the chunked formulas."""
    state = initial.copy()
    out = np.empty_like(v)
    for t in range(q.shape[1]):
        state = state * np.exp(g[:, t, :, None])
        residual = v[:, t] - np.einsum("hk,hkv->hv", k[:, t], state)
        state += k[:, t, :, None] * (beta[:, t, None] * residual)[:, None, :]
        out[:, t] = np.einsum("hk,hkv->hv", q[:, t], state)
    return out, state
