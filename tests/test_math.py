"""CPU-only algebra tests; no CUDA, no upstream checkout required."""

import numpy as np
import pytest

from flashkda_routes.routes.b_rescale import (
    bf16_numpy, chunk_core, recurrent_numpy,
)


def _inputs(heads=2, time=37, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(heads, time, dim))
    k = rng.normal(size=(heads, time, dim))
    v = rng.normal(size=(heads, time, dim))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    g = -rng.uniform(0, 0.5, size=(heads, time, dim))
    beta = rng.uniform(0.05, 0.95, size=(heads, time))
    return q, k, v, g, beta


@pytest.mark.parametrize("chunk", [16, 32, 64])
@pytest.mark.parametrize("time", [1, 16, 17, 37, 65])
def test_chunk_matches_token_recurrence(chunk, time):
    q, k, v, g, beta = _inputs(time=time)
    state = np.random.default_rng(1).normal(size=(2, 16, 16)) / 8
    ref_o, ref_s = recurrent_numpy(q, k, v, g, beta, state.copy())
    out, final = chunk_core(q, k, v, g, beta, state.copy(), chunk)
    np.testing.assert_allclose(out, ref_o, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(final, ref_s, atol=2e-12, rtol=2e-12)


def test_chunk_does_not_mutate_initial_state():
    q, k, v, g, beta = _inputs()
    state = np.random.default_rng(2).normal(size=(2, 16, 16))
    frozen = state.copy()
    chunk_core(q, k, v, g, beta, state, 32)
    np.testing.assert_array_equal(state, frozen)


def test_value_columns_are_separable():
    q, k, _, g, beta = _inputs()
    v = np.random.default_rng(3).normal(size=(2, q.shape[1], 32))
    state = np.random.default_rng(4).normal(size=(2, 16, 32))
    out_full, state_full = chunk_core(q, k, v, g, beta, state, 16)
    out_left, state_left = chunk_core(q, k, v[:, :, :16], g, beta, state[:, :, :16], 16)
    out_right, state_right = chunk_core(q, k, v[:, :, 16:], g, beta, state[:, :, 16:], 16)
    np.testing.assert_allclose(out_full, np.concatenate([out_left, out_right], axis=-1),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state_full, np.concatenate([state_left, state_right], axis=-1),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("chunk,alpha,expected_finite", [
    (16, 2.5, True), (16, 5.0, True), (32, 5.0, False), (64, 2.5, False),
])
def test_absolute_prefix_control_matches_range_boundary(chunk, alpha, expected_finite):
    q = np.ones((1, chunk, 16), np.float32) / 16
    k = np.ones_like(q) / 4
    v = np.ones_like(q)
    g = np.full_like(q, -alpha)
    beta = np.full((1, chunk), 0.5, np.float32)
    state = np.zeros((1, 16, 16), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        bad_o, bad_s = chunk_core(q, k, v, g, beta, state, chunk, rescale=False)
    finite = bool(np.isfinite(bad_o).all() and np.isfinite(bad_s).all())
    assert finite is expected_finite
    good_o, good_s = chunk_core(q, k, v, g, beta, state, chunk)
    assert np.isfinite(good_o).all() and np.isfinite(good_s).all()


def test_bf16_rounding_is_round_to_nearest_even():
    exact = np.array([1.0, 1.0078125, -2.5], dtype=np.float32)
    assert np.all(bf16_numpy(exact) == exact)
    # 1 + 2^-8 is exactly halfway between two BF16 values; ties round to even,
    # so the result is 1.0, not 1.0078125.
    halfway = np.float32(1.0 + 2 ** -8)
    assert bf16_numpy(halfway) == np.float32(1.0)
    # Just above the tie rounds up.
    above = np.float32(1.0 + 2 ** -8 + 2 ** -20)
    assert bf16_numpy(above) == np.float32(1.0078125)
