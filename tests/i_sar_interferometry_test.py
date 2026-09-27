"""Tests of i.sar.interferometry on simulated Sentinel-1 TOPS pairs."""

import os

import numpy as np
import pytest

import grass.script as gs

from conftest import description, read_map, run_module, target_env, wrap


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


def reference_lonlat(pair, sim):
    """A stable point: a corner of the image, far from the bowl."""
    g = pair.ref.pixel_ground(np.array([0]), np.array([8.0]), np.array([88.0]))
    lat, lon, _h = sim.ecef_to_geodetic(g)
    return float(lon[0]), float(lat[0])


def radar_truth(project, pair, sim, name):
    """True vertical motion at every pixel of a debursted product."""
    segments = description(project, name)["raster_geometry"]["segments"]
    rows = sum(s["rows"] for s in segments)
    truth = np.full((rows, sim.NSAMPLES), np.nan)
    for seg in segments:
        lines, samples = np.mgrid[
            seg["first_line"] : seg["first_line"] + seg["rows"], 0 : sim.NSAMPLES
        ]
        g = pair.ref.pixel_ground(
            np.full(lines.size, seg["burst"] - 1),
            lines.ravel().astype(float),
            samples.ravel().astype(float),
        )
        lat, lon, _h = sim.ecef_to_geodetic(g)
        truth[seg["first_row"] : seg["first_row"] + seg["rows"]] = pair.motion(
            lat, lon
        ).reshape(lines.shape)
    return truth


@pytest.fixture(scope="module")
def subsidence(project, sim):
    """Unwrapped subsidence of the defo pair, in radar and UTM geometry."""
    pair, _dem = project.pairs["defo"]
    lon, lat = reference_lonlat(pair, sim)
    proc = run_module(
        project,
        reference="defo_ref_iw1_vv",
        secondary="defo_co",
        output="defo",
        measure="phase,coherence,unwrapped_phase,los_displacement,vertical_displacement,"
        "incidence_angle,unwrap_component",
        reference_point=f"{lon},{lat}",
        target="utm40",
        resolution=10,
    )
    assert proc.returncode == 0, proc.stderr
    offset = pair.motion(np.array([lat]), np.array([lon]))[0]
    return pair, offset, proc


def test_subsidence_radar_geometry(project, sim, subsidence):
    pair, offset, _proc = subsidence
    vertical = read_map(project, "defo_vertical_displacement")
    truth = radar_truth(project, pair, sim, "defo_phase") - offset
    # Enough motion to need unwrapping: more than one fringe.
    assert truth.min() < -0.05
    error = (vertical - truth)[4:-4, 6:-6]
    assert np.nanmean(np.abs(error)) < 0.002
    assert np.sqrt(np.nanmean(error**2)) < 0.003
    los = read_map(project, "defo_los_displacement")
    incidence = read_map(project, "defo_incidence_angle")
    assert 30 < np.nanmean(incidence) < 45
    np.testing.assert_allclose(
        vertical, los / np.cos(np.radians(incidence)), rtol=1e-5, equal_nan=True
    )
    # Subsidence moves the ground away from the satellite: negative LOS.
    assert np.nanmin(los) < -0.04
    assert np.nanmax(read_map(project, "defo_unwrap_component")) == 1
    meta = description(project, "defo_vertical_displacement")["interferometry"]
    assert meta["unwrapping"]["components"] == 1
    assert meta["unwrapping"]["reference_pixel"] is not None


def test_subsidence_geocoded(project, sim, subsidence):
    from osgeo import osr

    pair, offset, _proc = subsidence
    env = target_env(project)
    geocoded = read_map(project, "defo_vertical_displacement", env=env)
    region = gs.parse_command(
        "g.region", flags="g", raster="defo_vertical_displacement", env=env
    )
    north, west, res = float(region["n"]), float(region["w"]), float(region["nsres"])
    assert res == 10
    rows, cols = np.nonzero(np.isfinite(geocoded))
    assert rows.size > 1000
    src, dst = osr.SpatialReference(), osr.SpatialReference()
    src.ImportFromEPSG(32640)
    dst.ImportFromEPSG(4326)
    for srs in (src, dst):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_geo = osr.CoordinateTransformation(src, dst)

    def truth(shift_e, shift_n):
        east = west + (cols + 0.5) * res + shift_e
        northing = north - (rows + 0.5) * res + shift_n
        pts = np.array(
            to_geo.TransformPoints(np.column_stack([east, northing]).tolist())
        )
        return pair.motion(pts[:, 1], pts[:, 0]) - offset

    error = geocoded[rows, cols] - truth(0.0, 0.0)
    assert np.sqrt(np.mean(error**2)) < 0.003
    # Geolocation: the best-fitting shift of the truth is below 5 m.
    shifts = [
        (de, dn)
        for de in (-10.0, -5.0, 0.0, 5.0, 10.0)
        for dn in (-10.0, -5.0, 0.0, 5.0, 10.0)
    ]
    rms = [
        np.sqrt(np.mean((geocoded[rows, cols] - truth(de, dn)) ** 2))
        for de, dn in shifts
    ]
    best = shifts[int(np.argmin(rms))]
    assert abs(best[0]) <= 5
    assert abs(best[1]) <= 5
    meta = description_in(project, env, "defo_vertical_displacement")
    assert meta["geocoding"]["method"] == "range-Doppler terrain correction"
    assert (
        gs.raster_info("defo_coherence", env=env)["semantic_label"]
        == "S1_VV_IFG_COHERENCE"
    )


def description_in(project, env, name):
    import json

    genv = gs.gisenv(env=env)
    path = os.path.join(
        genv["GISDBASE"],
        genv["LOCATION_NAME"],
        genv["MAPSET"],
        "cell_misc",
        name,
        "description.json",
    )
    with open(path) as fd:
        return json.load(fd)


def test_unwrap_mask(project):
    proc = run_module(
        project,
        reference="defo_ref_iw1_vv",
        secondary="defo_co",
        output="masked",
        measure="coherence,unwrapped_phase",
        unwrap_mask=0.999,
    )
    assert proc.returncode == 0, proc.stderr
    coh = read_map(project, "masked_coherence")
    unw = read_map(project, "masked_unwrapped_phase")
    assert np.all(np.isnan(unw[coh < 0.999]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"measure": "unwrapped_phase", "reference_point": "10,10"},
            "outside the image",
        ),
        ({"target": "nowhere"}, "not found"),
        ({"resolution": 10}, "target"),
    ],
)
def test_unwrap_failures(project, kwargs, message):
    proc = run_module(
        project,
        reference="defo_ref_iw1_vv",
        secondary="defo_co",
        output="fail",
        **kwargs,
    )
    assert proc.returncode != 0
    assert message in proc.stderr


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
