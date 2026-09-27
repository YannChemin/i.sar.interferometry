"""Fixtures for i.sar.interferometry tests.

Simulated Sentinel-1 TOPS pairs (the s1sim simulator of the
i.sar.coregistration tests) are imported with r.in.s1slc and coregistered
with i.sar.coregistration, both run from their source trees:

- I_SAR_COREGISTRATION: source directory of i.sar.coregistration
  (default ~/dev/i.sar.coregistration),
- R_IN_S1SLC: r.in.s1slc script (default ~/dev/r.in.s1slc/r.in.s1slc.py).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import grass.script as gs

HERE = Path(__file__).resolve().parent
SCRIPT = str(HERE.parent / "i.sar.interferometry.py")
COREG = Path(
    os.environ.get("I_SAR_COREGISTRATION", Path.home() / "dev" / "i.sar.coregistration")
)
R_IN_S1SLC = Path(
    os.environ.get("R_IN_S1SLC", Path.home() / "dev" / "r.in.s1slc" / "r.in.s1slc.py")
)

if (COREG / "tests" / "s1sim.py").is_file():
    sys.path.insert(0, str(COREG / "tests"))
    import s1sim  # noqa: E402
else:
    s1sim = None


@pytest.fixture(scope="session")
def sim():
    if s1sim is None or not R_IN_S1SLC.is_file():
        pytest.skip(
            "i.sar.coregistration or r.in.s1slc sources not found (set I_SAR_COREGISTRATION, R_IN_S1SLC)"
        )
    return s1sim


@pytest.fixture(scope="session")
def library(tmp_path_factory, sim):
    build = tmp_path_factory.mktemp("build")
    source = (COREG / "sarcoreg_core.h").read_text() + (
        COREG / "sarcoreg_kernels.cl"
    ).read_text()
    lines = [
        '"' + line.replace("\\", "\\\\").replace('"', '\\"') + '\\n"'
        for line in source.splitlines()
    ]
    (build / "sarcoreg_cl.h").write_text("\n".join(lines) + "\n")
    lib = build / "libsarcoreg.so"
    subprocess.run(
        [
            "gcc",
            "-O3",
            "-std=gnu11",
            "-fPIC",
            "-fopenmp",
            "-shared",
            f"-I{build}",
            f"-I{COREG}",
            "-o",
            str(lib),
            str(COREG / "sarcoreg.c"),
            "-lOpenCL",
            "-lm",
        ],
        check=True,
    )
    return lib


def run(session, script, *flags, env_extra=None, **kwargs):
    args = (
        [sys.executable, str(script)]
        + ["-" + f for f in flags]
        + [f"{k}={v}" for k, v in kwargs.items()]
    )
    env = dict(session.env)
    env.update(env_extra or {})
    return subprocess.run(args, env=env, capture_output=True, text=True, check=False)


@pytest.fixture(scope="session")
def project(tmp_path_factory, sim, library):
    """XY project with two coregistered pairs:

    - flat: 150 m baseline, flat terrain, burst maps (flat_ref_iw1_vv,
      flat_co) and debursted reference (deb_ref_iw1_vv);
    - hill: 1 km baseline over an 800 m hill, burst maps coregistered with
      the DEM and its elevation maps (hill_ref_iw1_vv, hill_co).
    """
    pairs = {}
    for name, (baseline, hill) in {
        "flat": (150.0, 0.0),
        "hill": (1000.0, 800.0),
    }.items():
        parent = tmp_path_factory.mktemp(name)
        pair = sim.Pair(parent, baseline=baseline, terrain_height=hill)
        dem = sim.write_dem(parent / "dem.tif", hill, pair.bbox()) if hill else None
        pairs[name] = (pair, dem)
    path = tmp_path_factory.mktemp("grassdata") / "xy"
    gs.create_project(path)
    with gs.setup.init(path, env=os.environ.copy()) as session:
        coreg_env = {"I_SAR_COREGISTRATION_LIB": str(library)}
        for name, (pair, dem) in pairs.items():
            for role, safe in (("ref", pair.ref_safe), ("sec", pair.sec_safe)):
                proc = run(
                    session, R_IN_S1SLC, "b", input=safe, output=f"{name}_{role}"
                )
                assert proc.returncode == 0, proc.stderr
            extra = (
                {"dem": dem, "dem_height": "ellipsoid", "extra": "elevation"}
                if dem
                else {}
            )
            proc = run(
                session,
                COREG / "i.sar.coregistration.py",
                env_extra=coreg_env,
                reference=f"{name}_ref_iw1_vv",
                secondary=f"{name}_sec_iw1_vv",
                output=f"{name}_co",
                device="host",
                **extra,
            )
            assert proc.returncode == 0, proc.stderr
        proc = run(
            session, R_IN_S1SLC, input=pairs["flat"][0].ref_safe, output="deb_ref"
        )
        assert proc.returncode == 0, proc.stderr
        session.pairs = pairs
        yield session


def run_module(session, *flags, **kwargs):
    return run(session, SCRIPT, *flags, **kwargs)


def read_map(session, name):
    import grass.script.array as garray

    env = dict(session.env)
    env["GRASS_REGION"] = gs.region_env(raster=name, env=env)
    return np.array(
        garray.array(name, null="nan", dtype=np.float32, env=env), dtype=np.float64
    )


def description(session, name):
    env = gs.gisenv(env=session.env)
    path = os.path.join(
        env["GISDBASE"],
        env["LOCATION_NAME"],
        env["MAPSET"],
        "cell_misc",
        name,
        "description.json",
    )
    with open(path) as fd:
        return json.load(fd)


def wrap(phase):
    return np.angle(np.exp(1j * phase))
