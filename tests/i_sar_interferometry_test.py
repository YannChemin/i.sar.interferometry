"""Tests of i.sar.interferometry on simulated Sentinel-1 TOPS pairs."""

import os

import numpy as np
import pytest

import grass.script as gs

from conftest import description, read_map, run_module, wrap


def circular_std(phase):
    z = np.exp(1j * phase[np.isfinite(phase)])
    return float(np.sqrt(-2 * np.log(np.abs(z.mean()))))


def test_debursted_interferogram(project):
    proc = run_module(
        project, reference="flat_ref_iw1_vv", secondary="flat_co", output="ifg"
    )
    assert proc.returncode == 0, proc.stderr
    phase = read_map(project, "ifg_phase")
    coh = read_map(project, "ifg_coherence")
    # Same debursted geometry as r.in.s1slc gives for the reference.
    deb = description(project, "deb_ref_iw1_vv_i")["raster_geometry"]
    geom = description(project, "ifg_phase")["raster_geometry"]
    assert phase.shape == (deb["rows"], deb["cols"])
    assert geom["segments"] == deb["segments"]
    assert geom["first_line_time"] == deb["first_line_time"]
    # Flat terrain, no deformation: the flattened phase is zero.
    inner = phase[4:-4, 6:-6]
    assert abs(np.angle(np.nanmean(np.exp(1j * inner)))) < 0.02
    assert circular_std(inner) < 0.35
    assert np.nanmean(coh[4:-4, 6:-6]) > 0.98
    assert np.nanmax(coh) <= 1.0


def test_reference_phase_matches_truth(project):
    pair, _dem = project.pairs["flat"]
    proc = run_module(
        project,
        "b",
        reference="flat_ref_iw1_vv",
        secondary="flat_co",
        output="bur",
        measure="reference_phase",
    )
    assert proc.returncode == 0, proc.stderr
    for k in range(3):
        synthetic = read_map(project, f"bur_b{k + 1:02d}_reference_phase")
        lines, samples = np.mgrid[
            0 : synthetic.shape[0] : 5, 0 : synthetic.shape[1] : 5
        ]
        truth, _h = pair.reference_phase(
            k, lines.ravel(), samples.ravel(), topography=False
        )
        diff = wrap(synthetic[lines, samples].ravel() - truth)
        assert np.max(np.abs(diff)) < 2e-3


def test_topography(project):
    common = {"reference": "hill_ref_iw1_vv", "secondary": "hill_co", "looks": "4,1"}
    for removal in ("flat_earth", "topography"):
        proc = run_module(
            project, output=f"hill_{removal}", phase_removal=removal, **common
        )
        assert proc.returncode == 0, proc.stderr
    flat = read_map(project, "hill_flat_earth_phase")
    topo = read_map(project, "hill_topography_phase")
    assert flat.shape == topo.shape
    # Topographic fringes remain after the flat-earth phase only.
    assert circular_std(flat[2:-2, 2:-2]) > 1.0
    assert circular_std(topo[2:-2, 2:-2]) < 0.35
    assert np.nanmean(read_map(project, "hill_topography_coherence")) > np.nanmean(
        read_map(project, "hill_flat_earth_coherence")
    )


def test_topography_needs_elevation(project):
    proc = run_module(
        project,
        reference="flat_ref_iw1_vv",
        secondary="flat_co",
        output="noelev",
        phase_removal="topography",
    )
    assert proc.returncode != 0
    assert "extra=elevation" in proc.stderr


def test_multilook_and_filter(project):
    proc = run_module(
        project,
        reference="hill_ref_iw1_vv",
        secondary="hill_co",
        output="ml",
        phase_removal="topography",
        looks="4,2",
        measure="phase,coherence,complex,amplitude",
    )
    assert proc.returncode == 0, proc.stderr
    proc = run_module(
        project,
        reference="hill_ref_iw1_vv",
        secondary="hill_co",
        output="gf",
        phase_removal="topography",
        looks="4,2",
        filter_alpha=0.8,
        filter_size=32,
    )
    assert proc.returncode == 0, proc.stderr
    full = description(project, "hill_topography_phase")["raster_geometry"]
    ml = read_map(project, "ml_phase")
    # Both have 4 range looks; 2 azimuth looks halve the rows.
    assert ml.shape == (full["rows"] // 2, full["cols"])
    i, q = read_map(project, "ml_i"), read_map(project, "ml_q")
    np.testing.assert_allclose(np.arctan2(q, i), ml, atol=1e-5)
    np.testing.assert_allclose(
        np.hypot(i, q), read_map(project, "ml_amplitude"), rtol=1e-5
    )
    geom = description(project, "ml_phase")["raster_geometry"]
    assert geom["looks_range"] == 4
    assert geom["looks_azimuth"] == 2
    filtered = read_map(project, "gf_phase")
    assert circular_std(filtered[2:-2, 2:-2]) < 0.5 * circular_std(ml[2:-2, 2:-2])
    assert np.array_equal(np.isnan(filtered), np.isnan(ml))


def test_metadata_and_group(project):
    proc = run_module(
        project, reference="flat_ref_iw1_vv", secondary="flat_co", output="meta"
    )
    assert proc.returncode == 0, proc.stderr
    meta = description(project, "meta_coherence")
    insar = meta["interferometry"]
    assert abs(insar["baselines"]["perpendicular"]) == pytest.approx(150.0, rel=0.02)
    assert insar["baselines"]["temporal"] == pytest.approx(12.0, abs=0.01)
    assert insar["coregistration"]["esd"]["applied"]
    assert meta["secondary"]["product"]["absolute_orbit_start"] == "46927"
    info = gs.raster_info("meta_phase", env=project.env)
    assert info["semantic_label"] == "S1_VV_IFG_PHASE"
    assert info["units"] == "radians"
    stamp = gs.read_command("r.timestamp", map="meta_phase", env=project.env)
    assert stamp.startswith("12 Jan 2023")
    assert "24 Jan 2023" in stamp
    env = gs.gisenv(env=project.env)
    group = os.path.join(
        env["GISDBASE"], env["LOCATION_NAME"], env["MAPSET"], "group", "meta"
    )
    points = np.loadtxt(os.path.join(group, "POINTS"), comments="#")
    assert points.shape[1] == 5
    assert points.shape[0] > 10


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"secondary": "flat_sec_iw1_vv"}, "i.sar.coregistration"),
        ({"secondary": "hill_co"}, "coregistered on"),
        ({"secondary": "flat_co", "looks": "0,1"}, "looks"),
    ],
)
def test_failures(project, kwargs, message):
    proc = run_module(project, reference="flat_ref_iw1_vv", output="fail", **kwargs)
    assert proc.returncode != 0
    assert message in proc.stderr
