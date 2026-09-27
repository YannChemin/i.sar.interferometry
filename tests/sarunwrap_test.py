"""Tests of libsarunwrap, the minimum cost flow phase unwrapping library."""

import ctypes

import numpy as np
import pytest


class Job(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
        ("phase", ctypes.POINTER(ctypes.c_float)),
        ("coherence", ctypes.POINTER(ctypes.c_float)),
        ("looks", ctypes.c_float),
        ("unwrapped", ctypes.POINTER(ctypes.c_float)),
        ("component", ctypes.POINTER(ctypes.c_int32)),
        ("stats", ctypes.c_int64 * 4),
    ]


def unwrap(library, phase, coherence, device="host", looks=4):
    lib = ctypes.CDLL(str(library))
    lib.sarunwrap_init.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    msg = ctypes.create_string_buffer(4096)
    if lib.sarunwrap_init(device.encode(), b"", 0, msg, 4096):
        pytest.skip(msg.value.decode())
    phase = np.ascontiguousarray(phase, dtype=np.float32)
    coherence = np.ascontiguousarray(coherence, dtype=np.float32)
    out = np.empty_like(phase)
    comp = np.empty(phase.shape, dtype=np.int32)
    job = Job(
        rows=phase.shape[0],
        cols=phase.shape[1],
        phase=phase.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        coherence=coherence.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        looks=looks,
        unwrapped=out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        component=comp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert lib.sarunwrap_run(ctypes.byref(job), msg, 4096) == 0, msg.value
    lib.sarunwrap_finish()
    return out, comp, list(job.stats)


def scene(n=300, seed=3):
    """Bowl and ramp of about 10 cycles, a decorrelated patch and a hole."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    truth = 40 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05) + 25 * x
    coh = np.full(truth.shape, 0.9)
    patch = (slice(int(0.1 * n), int(0.25 * n)), slice(int(0.6 * n), int(0.9 * n)))
    coh[patch] = 0.1
    noisy = truth + rng.normal(size=truth.shape) * np.sqrt((1 - coh**2) / (8 * coh**2))
    noisy[patch] = rng.uniform(-np.pi, np.pi, noisy[patch].shape)
    phase = np.angle(np.exp(1j * noisy)).astype(np.float32)
    phase[int(0.6 * n) : int(0.7 * n), int(0.2 * n) : int(0.3 * n)] = np.nan
    return noisy, phase, coh


def cycles_off(unwrapped, truth, mask):
    """Fraction of masked pixels off by whole cycles from the most common offset."""
    k = np.round((unwrapped - truth)[mask] / (2 * np.pi))
    return float(np.mean(k != np.median(k)))


def test_unwraps_bowl_around_noise_and_hole(unwrap_library):
    truth, phase, coh = scene()
    out, comp, stats = unwrap(unwrap_library, phase, coh)
    assert stats[0] > 100  # residues from the decorrelated patch
    assert stats[1] == 1
    good = (coh > 0.5) & np.isfinite(phase)
    assert cycles_off(out, truth, good) == 0.0
    # The unwrapped phase is congruent with the wrapped one, null on nulls.
    valid = np.isfinite(phase)
    assert np.array_equal(np.isfinite(out), valid)
    np.testing.assert_allclose(
        np.angle(np.exp(1j * (out - phase)))[valid], 0, atol=1e-3
    )
    assert np.all(comp[valid] == 1)
    assert np.all(comp[~valid] == 0)


def test_opencl_matches_host(unwrap_library):
    _truth, phase, coh = scene(200, seed=7)
    host, _c, host_stats = unwrap(unwrap_library, phase, coh, "host")
    for device in ("gpu", "cpu"):
        try:
            ocl, _c, stats = unwrap(unwrap_library, phase, coh, device)
        except pytest.skip.Exception:
            continue
        assert stats == host_stats
        np.testing.assert_allclose(ocl, host, atol=1e-4, equal_nan=True)
        return
    pytest.skip("no OpenCL device")


def test_masked_band_splits_components(unwrap_library):
    rng = np.random.default_rng(5)
    n = 300
    y, x = np.mgrid[0:n, 0:n] / n
    truth = 60 * np.exp(-((x - 0.4) ** 2 + (y - 0.5) ** 2) / 0.03) + 30 * x * y
    band = np.abs(x - 0.7 - 0.1 * np.sin(8 * y)) < 0.03
    coh = np.where(band, 0.05, 0.8)
    phase = np.angle(np.exp(1j * truth)).astype(np.float32)
    phase[band] = np.nan
    out, comp, stats = unwrap(unwrap_library, phase, coh)
    assert stats[1] == 2
    for c in (1, 2):
        sel = comp == c
        assert cycles_off(out, truth, sel) == 0.0
    # Components are numbered here by scan order; both sides are covered.
    assert set(np.unique(comp[x < 0.5])) == {comp[0, 0]}


def test_integration_is_exact_without_residues(unwrap_library):
    y, x = np.mgrid[0:50, 0:80]
    truth = 0.3 * x + 0.2 * y
    phase = np.angle(np.exp(1j * truth)).astype(np.float32)
    out, _comp, stats = unwrap(unwrap_library, phase, np.ones_like(phase))
    assert stats[0] == 0
    np.testing.assert_allclose(out - out[0, 0], truth - truth[0, 0], atol=1e-4)
