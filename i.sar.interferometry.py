#!/usr/bin/env python3

# MODULE:    i.sar.interferometry
# AUTHOR(S): Yann Chemin
# PURPOSE:   Computes the interferogram, coherence and derived products of
#            a Sentinel-1 TOPS SLC pair: a reference image imported by
#            r.in.s1slc and a secondary image coregistered on it by
#            i.sar.coregistration (SNAP Interferogram, TOPSAR-Deburst,
#            Multilook and Goldstein Phase Filtering).
# COPYRIGHT: (C) 2026 by Yann Chemin and the GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

# %module
# % description: Computes the interferogram and coherence of a coregistered Sentinel-1 TOPS SLC pair.
# % keyword: imagery
# % keyword: SAR
# % keyword: radar
# % keyword: Sentinel-1
# % keyword: SLC
# % keyword: interferometry
# % keyword: InSAR
# % keyword: coherence
# %end

# %option
# % key: reference
# % type: string
# % required: yes
# % key_desc: basename
# % label: Basename of the reference image imported by r.in.s1slc
# % description: The reference of the coregistration, e.g. s1_20230112_iw2_vv
# %end

# %option
# % key: secondary
# % type: string
# % required: yes
# % key_desc: basename
# % label: Basename of the secondary image coregistered by i.sar.coregistration
# %end

# %option G_OPT_R_BASENAME_OUTPUT
# % required: yes
# % description: Basename of the output maps and imagery group
# %end

# %option
# % key: measure
# % type: string
# % required: no
# % multiple: yes
# % options: phase,coherence,amplitude,complex,reference_phase
# % answer: phase,coherence
# % label: Products to write
# % descriptions: phase;Interferometric phase (radians, wrapped);coherence;Coherence magnitude;amplitude;Interferogram amplitude |m s*|;complex;Interferogram real and imaginary parts (maps _i and _q);reference_phase;Removed flat-earth (and topographic) phase, wrapped
# % guisection: Output
# %end

# %option
# % key: phase_removal
# % type: string
# % required: no
# % options: none,flat_earth,topography
# % answer: flat_earth
# % label: Synthetic phase removed from the interferogram
# % descriptions: none;Raw interferogram;flat_earth;Phase of the WGS84 ellipsoid;topography;Phase of the terrain given by the elevation maps of i.sar.coregistration (extra=elevation)
# %end

# %option
# % key: coherence_window
# % type: integer
# % required: no
# % multiple: yes
# % key_desc: range,azimuth
# % answer: 10,3
# % label: Coherence estimation window (samples, lines)
# % guisection: Processing
# %end

# %option
# % key: looks
# % type: integer
# % required: no
# % multiple: yes
# % key_desc: range,azimuth
# % answer: 1,1
# % label: Number of looks (samples, lines)
# % description: The interferogram is averaged, the coherence averaged, over looks
# % guisection: Processing
# %end

# %option
# % key: filter_alpha
# % type: double
# % required: no
# % answer: 0
# % options: 0-1
# % label: Goldstein filter exponent (0: no filtering)
# % description: Applied to the (multilooked) interferogram; the coherence is not filtered
# % guisection: Filter
# %end

# %option
# % key: filter_size
# % type: integer
# % required: no
# % options: 32,64,128
# % answer: 64
# % label: Goldstein filter FFT block size
# % guisection: Filter
# %end

# %flag
# % key: b
# % label: Keep burst maps separate (do not deburst)
# % description: With burst inputs, writes one set of maps per burst
# %end

import atexit
import json
import os
import sys
from datetime import datetime, timedelta

import grass.script as gs

C = 299792458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Spacing of the geometry nodes (lines, samples) of the synthetic phase.
NODE_LINES = 10
NODE_SAMPLES = 100
# Height margin (meters) of the nodes around the elevation range.
HEIGHT_MARGIN = 50.0
# Goldstein filter: spectrum smoothing window.
FILTER_WINDOW = 3

TMP_FILES = []


def cleanup():
    for path in TMP_FILES:
        try:
            os.remove(path)
        except OSError:
            pass


def parse_time(text):
    """Parse an ISO UTC time with any number of fractional digits."""
    text = text.strip().rstrip("Z")
    if "." in text:
        head, frac = text.split(".")
        frac = (frac + "000000")[:6]
    else:
        head, frac = text, "000000"
    return datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(microsecond=int(frac))


class TimeAxis:
    """Seconds since a common epoch, exact to the microsecond."""

    def __init__(self, epoch):
        self.epoch = epoch

    def sec(self, text):
        return (parse_time(text) - self.epoch) // timedelta(microseconds=1) * 1e-6

    def iso(self, sec):
        return (self.epoch + timedelta(microseconds=round(sec * 1e6))).isoformat()

    def datetime(self, sec):
        return self.epoch + timedelta(microseconds=round(sec * 1e6))


# Geodesy and orbits (same algorithms as i.sar.coregistration).


def geodetic_to_ecef(lat, lon, h):
    import numpy as np

    lat = np.radians(lat)
    lon = np.radians(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack(
        [
            (n + h) * np.cos(lat) * np.cos(lon),
            (n + h) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - WGS84_E2) + h) * np.sin(lat),
        ],
        axis=-1,
    )


def ecef_to_geodetic(p):
    import numpy as np

    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    rho = np.hypot(x, y)
    lat = np.arctan2(z, rho * (1.0 - WGS84_E2))
    for _i in range(6):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        h = rho / np.cos(lat) - n
        lat = np.arctan2(z, rho * (1.0 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    h = rho / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


class Orbit:
    """Orbit state vectors with Lagrange interpolation (SNAP OrbitStateVectors)."""

    def __init__(self, times, pos, vel, npoints=8):
        import numpy as np

        order = np.argsort(times)
        self.t = np.asarray(times, dtype=np.float64)[order]
        self.pos = np.asarray(pos, dtype=np.float64)[order]
        self.vel = np.asarray(vel, dtype=np.float64)[order]
        if self.t.size < 2:
            gs.fatal(_("At least two orbit state vectors are needed"))
        self.m = min(npoints, self.t.size)

    def _weights(self, t):
        import numpy as np

        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        n, m = self.t.size, self.m
        i0 = np.clip(np.searchsorted(self.t, t) - m // 2, 0, n - m)
        idx = i0[:, None] + np.arange(m)
        tt = self.t[idx]
        num = np.broadcast_to((t[:, None] - tt)[:, None, :], (t.size, m, m)).copy()
        den = tt[:, :, None] - tt[:, None, :]
        eye = np.eye(m, dtype=bool)
        num[:, eye] = 1.0
        den[:, eye] = 1.0
        return idx, np.prod(num / den, axis=2)

    def state(self, t):
        import numpy as np

        idx, w = self._weights(t)
        return np.einsum("ij,ijk->ik", w, self.pos[idx]), np.einsum(
            "ij,ijk->ik", w, self.vel[idx]
        )

    def acceleration(self, t, h=0.01):
        return (self.state(t + h)[1] - self.state(t - h)[1]) / (2.0 * h)


def zero_doppler(orbit, points, t0):
    """Zero-Doppler time and slant range of ECEF points (Newton iterations)."""
    import numpy as np

    t = np.array(t0, dtype=np.float64)
    for _i in range(30):
        s, v = orbit.state(t)
        a = orbit.acceleration(t)
        d = points - s
        f = np.einsum("ij,ij->i", d, v)
        fp = -np.einsum("ij,ij->i", v, v) + np.einsum("ij,ij->i", d, a)
        step = -f / fp
        t = t + step
        if np.max(np.abs(step)) < 1e-11:
            break
    s, _v = orbit.state(t)
    return t, np.linalg.norm(points - s, axis=1)


def range_doppler_to_ecef(orbit, t, rng, h, guess):
    """Ground point at zero-Doppler time t, slant range rng and ellipsoidal
    height h (DORIS lph2xyz with geodetic height correction)."""
    import numpy as np

    s, v = orbit.state(t)
    p = np.array(guess, dtype=np.float64)
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), t.shape)
    dh = np.zeros_like(h)
    for _outer in range(3):
        ah = WGS84_A + h + dh
        bh = WGS84_B + h + dh
        for _i in range(12):
            d = p - s
            f = np.stack(
                [
                    np.einsum("ij,ij->i", d, v),
                    np.einsum("ij,ij->i", d, d) - rng**2,
                    (p[:, 0] ** 2 + p[:, 1] ** 2) / ah**2 + p[:, 2] ** 2 / bh**2 - 1.0,
                ],
                axis=1,
            )
            jac = np.stack(
                [
                    v,
                    2.0 * d,
                    np.stack(
                        [2 * p[:, 0] / ah**2, 2 * p[:, 1] / ah**2, 2 * p[:, 2] / bh**2],
                        1,
                    ),
                ],
                axis=1,
            )
            step = np.linalg.solve(jac, -f[:, :, None])[:, :, 0]
            p += step
            if np.max(np.abs(step)) < 1e-5:
                break
        dh += h - ecef_to_geodetic(p)[2]
    return p


# Maps and metadata.


def exists(name):
    return bool(gs.find_file(name, element="cell")["file"])


def map_meta(name, what):
    found = gs.find_file(name, element="cell")
    if not found["file"]:
        gs.fatal(_("Raster map <{}> not found").format(name))
    env = gs.gisenv()
    path = os.path.join(
        env["GISDBASE"],
        env["LOCATION_NAME"],
        found["mapset"],
        "cell_misc",
        found["name"],
        "description.json",
    )
    if not os.path.isfile(path):
        gs.fatal(_("Raster map <{}> has no {} metadata ({})").format(name, what, path))
    with open(path) as fd:
        meta = json.load(fd)
    for key in ("product", "swath", "raster_geometry"):
        if key not in meta:
            gs.fatal(_("Metadata of <{}> lack the '{}' section").format(name, key))
    if "segments" not in meta["raster_geometry"]:
        gs.fatal(
            _(
                "Metadata of <{}> have no row segments; re-import with a current r.in.s1slc"
            ).format(name)
        )
    return meta


def image_stems(base, role):
    """Map stems of an image: [base] if debursted, else its burst stems."""
    debursted = exists(base + "_i") and exists(base + "_q")
    bursts = sorted(
        m.split("@")[0]
        for m in gs.list_strings("raster", pattern=base + "_b[0-9][0-9]_i")
    )
    if debursted and bursts:
        gs.fatal(
            _(
                "Both debursted maps <{b}_i> and burst maps <{b}_bNN_i> exist ({r} image)"
            ).format(b=base, r=role)
        )
    if not debursted and not bursts:
        gs.fatal(
            _("No complex maps <{b}_i> or <{b}_bNN_i> found ({r} image)").format(
                b=base, r=role
            )
        )
    return (True, [base]) if debursted else (False, [m[: -len("_i")] for m in bursts])


def read_raster(name, rows, cols):
    import numpy as np

    path = gs.tempfile(create=False)
    TMP_FILES.append(path)
    env = os.environ.copy()
    env["GRASS_REGION"] = gs.region_env(raster=name)
    info = gs.region(env=env)
    if (int(info["rows"]), int(info["cols"])) != (rows, cols):
        gs.fatal(
            _("Raster map <{}> is {}x{}, expected {}x{}").format(
                name, info["rows"], info["cols"], rows, cols
            )
        )
    gs.run_command(
        "r.out.bin",
        flags="f",
        input=name,
        output=path,
        bytes=4,
        null="nan",
        quiet=True,
        env=env,
    )
    data = np.fromfile(path, dtype=np.float32).reshape(rows, cols)
    os.remove(path)
    TMP_FILES.remove(path)
    return data


def read_complex(stem, rows, cols):
    return read_raster(stem + "_i", rows, cols).astype("complex64") + 1j * read_raster(
        stem + "_q", rows, cols
    )


def write_raster(name, data, title):
    import numpy as np

    rows, cols = data.shape
    path = gs.tempfile(create=False)
    TMP_FILES.append(path)
    np.ascontiguousarray(data, dtype=np.float32).tofile(path)
    gs.run_command(
        "r.in.bin",
        flags="f",
        input=path,
        output=name,
        title=title,
        bytes=4,
        order="native",
        north=rows,
        south=0,
        east=cols,
        west=0,
        rows=rows,
        cols=cols,
        anull="nan",
        overwrite=gs.overwrite(),
        quiet=True,
    )
    os.remove(path)
    TMP_FILES.remove(path)


class Acquisition:
    """Orbit and timing of one image of the pair, from its swath metadata."""

    def __init__(self, swath, axis):
        import numpy as np

        self.swath = swath
        self.axis = axis
        self.dt = float(swath["azimuth_time_interval"])
        self.lpb = int(swath["lines_per_burst"])
        self.ns = int(swath["number_of_samples"])
        self.fs = float(swath["range_sampling_rate"])
        self.srt = float(swath["slant_range_time"])
        self.wavelength = float(swath["wavelength"])
        self.burst_t = np.array([axis.sec(b["azimuth_time"]) for b in swath["bursts"]])
        osv = swath["orbit_state_vectors"]
        if not osv:
            gs.fatal(_("No orbit state vectors in the metadata"))
        self.orbit = Orbit(
            [axis.sec(o["time"]) for o in osv],
            [o["position"] for o in osv],
            [o["velocity"] for o in osv],
        )
        self.first_line_time = axis.sec(swath["product_first_line_utc_time"])
        grid = swath["geolocation_grid"]
        times = sorted({axis.sec(p["azimuth_time"]) for p in grid})
        pixels = sorted({int(p["pixel"]) for p in grid})
        self.grid_t = np.array(times)
        self.grid_p = np.array(pixels, dtype=np.float64)
        shape = (len(times), len(pixels))
        self.grid = {
            k: np.full(shape, np.nan) for k in ("latitude", "longitude", "height")
        }
        ti = {t: i for i, t in enumerate(times)}
        pi = {p: j for j, p in enumerate(pixels)}
        for p in grid:
            i, j = ti[axis.sec(p["azimuth_time"])], pi[int(p["pixel"])]
            for k in self.grid:
                self.grid[k][i, j] = p[k]
        if len(times) < 2 or len(pixels) < 2 or np.isnan(self.grid["latitude"]).any():
            gs.fatal(_("Incomplete geolocation grid in the metadata"))

    def grid_interp(self, t, pixel):
        import numpy as np

        def weights(values, v):
            i = np.clip(np.searchsorted(values, v) - 1, 0, values.size - 2)
            return i, (v - values[i]) / (values[i + 1] - values[i])

        i, fi = weights(self.grid_t, t)
        j, fj = weights(self.grid_p, pixel)
        out = {}
        for k, g in self.grid.items():
            out[k] = (1 - fi) * ((1 - fj) * g[i, j] + fj * g[i, j + 1]) + fi * (
                (1 - fj) * g[i + 1, j] + fj * g[i + 1, j + 1]
            )
        return out

    def line_time(self, k, line):
        return self.burst_t[k] + line * self.dt

    def slant_range(self, sample):
        return 0.5 * C * (self.srt + sample / self.fs)

    def ground(self, t, sample, h):
        g = self.grid_interp(t, sample)
        guess = geodetic_to_ecef(g["latitude"], g["longitude"], h)
        return range_doppler_to_ecef(self.orbit, t, self.slant_range(sample), h, guess)


def synthetic_phase(ref, sec, burst, first_line, rows, cols, height):
    """Synthetic interferometric phase -4 pi (R_ref - R_sec) / lambda of
    reference burst lines first_line ... first_line + rows - 1.

    Computed at geometry nodes for the ellipsoid (height None) or for two
    heights enclosing the given height map, then interpolated bilinearly
    in lines and samples and linearly in height."""
    import numpy as np

    l1 = first_line + rows - 1
    lines = np.unique(np.r_[np.arange(first_line, l1, NODE_LINES), l1]).astype(
        np.float64
    )
    samples = np.unique(np.r_[np.arange(0, cols - 1, NODE_SAMPLES), cols - 1]).astype(
        np.float64
    )
    if lines.size < 2:
        lines = np.array([first_line - 0.5, first_line + 0.5])
    ll, ss = np.meshgrid(lines, samples, indexing="ij")
    t = ref.line_time(burst, ll.ravel())
    if height is None or not np.isfinite(height).any():
        heights = [0.0]
    else:
        heights = [
            float(np.nanmin(height)) - HEIGHT_MARGIN,
            float(np.nanmax(height)) + HEIGHT_MARGIN,
        ]
    t_guess = t - ref.first_line_time + sec.first_line_time
    drs = []
    for h in heights:
        p = ref.ground(t, ss.ravel(), h)
        _ts, r_sec = zero_doppler(sec.orbit, p, t_guess)
        r_ref = ref.slant_range(ss.ravel())
        drs.append((r_ref - r_sec).reshape(ll.shape))

    # Bilinear interpolation of the node values to every pixel.
    pl = np.arange(first_line, first_line + rows, dtype=np.float64)
    ps = np.arange(cols, dtype=np.float64)

    def weights(nodes, v):
        i = np.clip(np.searchsorted(nodes, v, side="right") - 1, 0, nodes.size - 2)
        return i, (v - nodes[i]) / (nodes[i + 1] - nodes[i])

    il, fl = weights(lines, pl)
    js, fs = weights(samples, ps)

    def interp(a):
        top = (1 - fs) * a[il][:, js] + fs * a[il][:, js + 1]
        bottom = (1 - fs) * a[il + 1][:, js] + fs * a[il + 1][:, js + 1]
        return (1 - fl)[:, None] * top + fl[:, None] * bottom

    dr = interp(drs[0])
    if len(heights) == 2:
        w = (height - heights[0]) / (heights[1] - heights[0])
        dr = dr + w * (interp(drs[1]) - dr)
    return -4.0 * np.pi * dr / ref.wavelength


def baselines(ref, sec, burst):
    """Perpendicular and parallel baselines (m), incidence angle and height
    of ambiguity at the centre of a reference burst."""
    import numpy as np

    t = np.array([ref.line_time(burst, 0.5 * (ref.lpb - 1))])
    sample = np.array([0.5 * (ref.ns - 1)])
    p = ref.ground(t, sample, 0.0)
    s_ref, _v = ref.orbit.state(t)
    ts, r_sec = zero_doppler(
        sec.orbit, p, t - ref.first_line_time + sec.first_line_time
    )
    s_sec, _v = sec.orbit.state(ts)
    los = (p - s_ref)[0]
    r_ref = np.linalg.norm(los)
    los /= r_ref
    b = (s_sec - s_ref)[0]
    b_par = float(np.dot(b, los))
    lat, lon, _h = ecef_to_geodetic(p)
    normal = geodetic_to_ecef(lat, lon, 1.0)[0] - geodetic_to_ecef(lat, lon, 0.0)[0]
    # Positive when the secondary orbit is above the reference one, in the
    # plane of the line of sight and the local vertical.
    away = normal - np.dot(normal, los) * los
    away /= np.linalg.norm(away)
    b_perp = float(np.dot(b - b_par * los, away))
    incidence = float(np.degrees(np.arccos(np.dot(-los, normal))))
    hoa = (
        ref.wavelength * r_ref * np.sin(np.radians(incidence)) / (2 * b_perp)
        if b_perp
        else float("inf")
    )
    return {
        "perpendicular": b_perp,
        "parallel": b_par,
        "incidence_angle": incidence,
        "height_of_ambiguity": float(hoa),
        "latitude": float(lat[0]),
        "longitude": float(lon[0]),
    }


# Signal processing.


def boxcar(a, wy, wx):
    """Moving average over a wy x wx window (edges shrink the window)."""
    import numpy as np

    def along(x, w, axis):
        x = np.moveaxis(x, axis, 0)
        c = np.cumsum(
            np.concatenate([np.zeros((1,) + x.shape[1:], x.dtype), x]), axis=0
        )
        n = x.shape[0]
        lo = np.clip(np.arange(n) - w // 2, 0, n)
        hi = np.clip(np.arange(n) - w // 2 + w, 0, n)
        out = (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))
        return np.moveaxis(out, 0, axis)

    return along(along(a, wy, 0), wx, 1)


def coherence(m, s_flat, window_rg, window_az):
    """|<m s*>| / sqrt(<|m|^2> <|s|^2>) over the window, s being the
    secondary with the synthetic phase applied (SNAP coherence3)."""
    import numpy as np

    valid = np.isfinite(m) & np.isfinite(s_flat)
    m0 = np.where(valid, m, 0).astype(np.complex128)
    s0 = np.where(valid, s_flat, 0).astype(np.complex128)
    num = boxcar(m0 * np.conj(s0), window_az, window_rg)
    den = np.sqrt(
        boxcar(np.abs(m0) ** 2, window_az, window_rg)
        * boxcar(np.abs(s0) ** 2, window_az, window_rg)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(den > 0, np.abs(num) / den, np.nan)
    return np.where(valid, np.minimum(gamma, 1.0), np.nan).astype(np.float32)


def multilook(a, looks_rg, looks_az):
    """Average over looks_az x looks_rg blocks, ignoring nulls; the last
    partial blocks are dropped."""
    import numpy as np

    if looks_rg == 1 and looks_az == 1:
        return a
    rows, cols = a.shape[0] // looks_az, a.shape[1] // looks_rg
    block = a[: rows * looks_az, : cols * looks_rg].reshape(
        rows, looks_az, cols, looks_rg
    )
    valid = np.isfinite(block)
    total = np.where(valid, block, 0).sum(axis=(1, 3))
    count = valid.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan).astype(a.dtype)


def goldstein(ifg, alpha, size):
    """Goldstein-Werner adaptive filter (SNAP GoldsteinFilterOp): blocks of
    size x size samples overlapping by 3/4, spectrum weighted by its
    3 x 3 smoothed magnitude to the power alpha, blocks recombined with
    triangular weights. Nulls stay null."""
    import numpy as np

    valid = np.isfinite(ifg)
    z = np.where(valid, ifg, 0).astype(np.complex128)
    rows, cols = z.shape
    if rows < size or cols < size:
        pad = ((0, max(0, size - rows)), (0, max(0, size - cols)))
        z = np.pad(z, pad)
    prow, pcol = z.shape
    out = np.zeros_like(z)
    weight = np.zeros(z.shape)
    step = size // 4
    tri = 1.0 - np.abs(np.arange(size) - size / 2.0 + 0.5) / (size / 2.0)
    wblock = np.outer(tri, tri)

    def origins(n):
        o = list(range(0, n - size + 1, step))
        if o[-1] != n - size:
            o.append(n - size)
        return o

    half = FILTER_WINDOW // 2
    for r in origins(prow):
        for c in origins(pcol):
            block = z[r : r + size, c : c + size]
            if not np.any(block):
                continue
            spec = np.fft.fft2(block)
            power = np.abs(spec)
            # Mean of the non-zero magnitudes in the window, circularly.
            padded = np.pad(power, half, mode="wrap")
            nonzero = np.pad((power != 0).astype(float), half, mode="wrap")
            ssum = sum(
                padded[i : i + size, j : j + size]
                for i in range(FILTER_WINDOW)
                for j in range(FILTER_WINDOW)
            )
            scount = sum(
                nonzero[i : i + size, j : j + size]
                for i in range(FILTER_WINDOW)
                for j in range(FILTER_WINDOW)
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                smooth = np.where(scount > 0, ssum / scount, 0.0)
            filtered = np.fft.ifft2(spec * smooth**alpha)
            out[r : r + size, c : c + size] += filtered * wblock
            weight[r : r + size, c : c + size] += wblock
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(weight > 0, out / weight, 0)
    out = out[:rows, :cols]
    return np.where(valid, out, np.nan + 1j * np.nan).astype(np.complex64)


def deburst_rows(acq, bursts):
    """(burst, line) of each debursted output row, SNAP TOPSAR-Deburst rule
    as in r.in.s1slc: rows regularly sampled in azimuth time from the first
    valid line of the first burst to the last valid line of the last one,
    the burst switching in the middle of each overlap."""
    import numpy as np

    sw = acq.swath["bursts"]
    t0 = acq.line_time(bursts[0], sw[bursts[0]]["first_valid_line"])
    t1 = acq.line_time(bursts[-1], sw[bursts[-1]]["last_valid_line"])
    n = int(round((t1 - t0) / acq.dt)) + 1
    times = t0 + np.arange(n) * acq.dt
    owner = np.full(n, bursts[0])
    for a, b in zip(bursts[:-1], bursts[1:]):
        mid = 0.5 * (acq.line_time(a, acq.lpb - 1) + acq.burst_t[b])
        owner[times > mid] = b
    line = np.floor((times - acq.burst_t[owner]) / acq.dt + 0.5).astype(np.int64)
    return owner, line, times


def segments_of(owner, line, bursts_meta):
    runs = []
    for r, (k, lb) in enumerate(zip(owner.tolist(), line.tolist())):
        last = runs[-1] if runs else None
        if last and last["burst"] == k + 1 and last["first_line"] + last["rows"] == lb:
            last["rows"] += 1
        else:
            runs.append(
                {
                    "burst": k + 1,
                    "burst_id": bursts_meta[k].get("burst_id"),
                    "first_row": r,
                    "rows": 1,
                    "first_line": lb,
                }
            )
    return runs


def write_gcps(group, acq, times, rows, looks_az, looks_rg, cols):
    """Imagery group POINTS from the reference geolocation grid, for
    i.rectify (image coordinates of cell centres, as r.in.s1slc)."""
    import numpy as np

    dt = acq.dt * looks_az
    t0 = times[0] + 0.5 * (looks_az - 1) * acq.dt
    r = (acq.grid_t - t0) / dt
    r = r[(r >= 0) & (r <= rows - 1)]
    r = np.unique(np.concatenate([[0.0, rows - 1.0], r]))
    pixels = acq.grid_p
    c = (pixels - 0.5 * (looks_rg - 1)) / looks_rg
    keep = (c >= 0) & (c <= cols - 1)
    pixels, c = pixels[keep], c[keep]
    tt = np.repeat(t0 + r * dt, pixels.size)
    pp = np.tile(pixels, r.size)
    g = acq.grid_interp(tt, pp)
    env = gs.gisenv()
    gdir = os.path.join(
        env["GISDBASE"], env["LOCATION_NAME"], env["MAPSET"], "group", group
    )
    with open(os.path.join(gdir, "POINTS"), "w") as fd:
        fd.write("# %7s %15s %15s %15s %9s status\n" % ("", "image", "", "target", ""))
        fd.write(
            "# %15s %15s %15s %15s   (1=ok)\n" % ("east", "north", "east", "north")
        )
        fd.write("#\n")
        for e1, n1, e2, n2 in zip(
            np.tile(c + 0.5, r.size),
            np.repeat(rows - (r + 0.5), c.size),
            g["longitude"],
            g["latitude"],
        ):
            fd.write("  %15f %15f %15f %15f %4d\n" % (e1, n1, e2, n2, 1))
    return r.size * c.size


def main():
    import numpy as np

    options, flags = gs.parser()
    atexit.register(cleanup)

    if int(gs.region()["projection"]) != 0:
        gs.fatal(
            _(
                "SLC images are in radar geometry: the current project must be unprojected (XY)"
            )
        )

    measures = options["measure"].split(",") if options["measure"] else []
    if not measures:
        gs.fatal(_("Nothing to compute: select at least one measure"))
    removal = options["phase_removal"]
    win = [int(v) for v in options["coherence_window"].split(",")]
    looks = [int(v) for v in options["looks"].split(",")]
    if len(win) != 2 or min(win) < 1:
        gs.fatal(_("coherence_window must be two positive integers (range,azimuth)"))
    if len(looks) != 2 or min(looks) < 1:
        gs.fatal(_("looks must be two positive integers (range,azimuth)"))
    alpha = float(options["filter_alpha"])
    fsize = int(options["filter_size"])

    ref_base, sec_base = options["reference"], options["secondary"]
    ref_deb, ref_stems = image_stems(ref_base, "reference")
    sec_deb, sec_stems = image_stems(sec_base, "secondary")
    if ref_deb != sec_deb:
        gs.fatal(_("Reference and secondary must both be debursted or both burst maps"))

    # Pair the maps: the coregistered secondary has the reference geometry
    # and the reference burst tags.
    pairs = []
    for s_stem in sec_stems:
        tag = s_stem[len(sec_base) :]
        r_stem = ref_base + tag
        s_meta = map_meta(s_stem + "_i", "i.sar.coregistration")
        co = s_meta.get("coregistration")
        if not co:
            gs.fatal(
                _("<{}> was not produced by i.sar.coregistration").format(s_stem + "_i")
            )
        if r_stem + "_i" not in [m.split("@")[0] for m in co["reference_maps"]]:
            gs.fatal(
                _("<{}> was coregistered on <{}>, not on <{}>").format(
                    s_stem + "_i", co["reference_maps"][0], r_stem + "_i"
                )
            )
        pairs.append((r_stem, s_stem, map_meta(r_stem + "_i", "r.in.s1slc"), s_meta))
    missing = sorted(set(ref_stems) - {p[0] for p in pairs})
    if missing:
        gs.message(
            _("Reference maps without coregistered secondary, ignored: {}").format(
                ", ".join(missing)
            )
        )

    r_meta0, s_meta0 = pairs[0][2], pairs[0][3]
    epoch = parse_time(r_meta0["swath"]["product_first_line_utc_time"]).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    axis = TimeAxis(epoch)
    ref = Acquisition(r_meta0["swath"], axis)
    sec = Acquisition(s_meta0["swath"], axis)

    elevation_stems = {}
    if removal == "topography":
        for _r, s_stem, _rm, _sm in pairs:
            name = s_stem + "_elevation"
            if not exists(name):
                gs.fatal(
                    _(
                        "Topographic phase removal needs <{}>: run i.sar.coregistration "
                        "with extra=elevation (and a dem)"
                    ).format(name)
                )
            elevation_stems[s_stem] = name

    deburst = not ref_deb and not flags["b"]
    prefix = options["output"]
    outputs = []
    if deburst or ref_deb:
        outputs.append(prefix)
    else:
        outputs += [prefix + s[len(sec_base) :] for _r, s, _a, _b in pairs]
    suffixes = []
    for m in measures:
        suffixes += ["i", "q"] if m == "complex" else [m]
    if not gs.overwrite():
        for stem in outputs:
            for s in suffixes:
                if gs.find_file("{}_{}".format(stem, s), element="cell", mapset=".")[
                    "file"
                ]:
                    gs.fatal(
                        _(
                            "Raster map <{}_{}> already exists, use --overwrite to replace it"
                        ).format(stem, s)
                    )

    info = baselines(
        ref,
        sec,
        int(pairs[len(pairs) // 2][2]["raster_geometry"]["segments"][0]["burst"]) - 1,
    )
    days = (
        axis.datetime(sec.first_line_time) - axis.datetime(ref.first_line_time)
    ).total_seconds() / 86400.0
    info["temporal"] = days
    gs.message(
        _(
            "Baselines: perpendicular {:.1f} m, parallel {:.1f} m, temporal {:.1f} days; "
            "height of ambiguity {:.1f} m"
        ).format(
            info["perpendicular"], info["parallel"], days, info["height_of_ambiguity"]
        )
    )

    # Full-resolution interferogram, coherence and synthetic phase per map.
    results = []
    for r_stem, s_stem, r_meta, s_meta in pairs:
        geom = r_meta["raster_geometry"]
        rows, cols = geom["rows"], geom["cols"]
        gs.message(_("Interferogram of <{}> and <{}>...").format(r_stem, s_stem))
        m = read_complex(r_stem, rows, cols)
        s = read_complex(s_stem, rows, cols)
        height = None
        if removal == "topography":
            height = read_raster(elevation_stems[s_stem], rows, cols).astype(np.float64)
        phase = np.zeros((rows, cols))
        if removal != "none":
            for seg in geom["segments"]:
                r0, n = seg["first_row"], seg["rows"]
                h = None if height is None else height[r0 : r0 + n]
                phase[r0 : r0 + n] = synthetic_phase(
                    ref, sec, seg["burst"] - 1, seg["first_line"], n, cols, h
                )
            if height is not None:
                phase[~np.isfinite(height)] = np.nan
        s_flat = s * np.exp(1j * phase)
        ifg = (m * np.conj(s_flat)).astype(np.complex64)
        coh = coherence(m, s_flat, win[0], win[1]) if "coherence" in measures else None
        results.append(
            {
                "geom": geom,
                "meta": r_meta,
                "s_meta": s_meta,
                "ifg": ifg,
                "coh": coh,
                "phase": phase,
            }
        )

    # Deburst the burst results into one image.
    if deburst:
        bursts = [r["geom"]["segments"][0]["burst"] - 1 for r in results]
        order = np.argsort(bursts)
        results = [results[i] for i in order]
        bursts = sorted(bursts)
        if bursts != list(range(bursts[0], bursts[-1] + 1)):
            gs.fatal(
                _(
                    "Bursts {} are not contiguous; debursting needs a contiguous range (or use -b)"
                ).format(",".join(str(k + 1) for k in bursts))
            )
        owner, line, times = deburst_rows(ref, bursts)
        cols = results[0]["geom"]["cols"]
        merged = {
            "ifg": np.full(
                (owner.size, cols), np.nan + 1j * np.nan, dtype=np.complex64
            ),
            "coh": np.full((owner.size, cols), np.nan, dtype=np.float32)
            if "coherence" in measures
            else None,
            "phase": np.full((owner.size, cols), np.nan),
        }
        for res, k in zip(results, bursts):
            rows_k = np.flatnonzero(owner == k)
            src = line[rows_k]
            ok = (src >= 0) & (src < res["ifg"].shape[0])
            for key in merged:
                if merged[key] is not None:
                    merged[key][rows_k[ok]] = res[key][src[ok]]
        bursts_meta = ref.swath["bursts"]
        geom = dict(results[0]["geom"])
        geom.update(
            {
                "debursted": True,
                "bursts": [k + 1 for k in bursts],
                "rows": int(owner.size),
                "first_line_time": axis.iso(times[0]),
                "last_line_time": axis.iso(times[-1]),
                "segments": segments_of(owner, line, bursts_meta),
            }
        )
        geom.pop("source_first_line", None)
        merged.update(
            {
                "geom": geom,
                "meta": results[0]["meta"],
                "s_meta": results[0]["s_meta"],
                "times": times,
            }
        )
        results = [merged]
        stems_out = [prefix]
    else:
        for res in results:
            geom = res["geom"]
            t0 = axis.sec(geom["first_line_time"])
            res["times"] = t0 + np.arange(geom["rows"]) * ref.dt
        stems_out = outputs

    for res, stem in zip(results, stems_out):
        write_products(
            options,
            res,
            stem,
            measures,
            removal,
            win,
            looks,
            alpha,
            fsize,
            info,
            ref,
            sec,
            axis,
        )
    return 0


def write_products(
    options,
    res,
    stem,
    measures,
    removal,
    win,
    looks,
    alpha,
    fsize,
    info,
    ref,
    sec,
    axis,
):
    import numpy as np

    ifg = multilook(res["ifg"], looks[0], looks[1])
    coh = multilook(res["coh"], looks[0], looks[1]) if res["coh"] is not None else None
    phase = res["phase"]
    if "reference_phase" in measures:
        ref_phase = np.angle(
            multilook(np.exp(1j * phase).astype(np.complex64), looks[0], looks[1])
        )
    if alpha > 0:
        gs.message(
            _("Goldstein filtering (alpha {}, blocks of {})...").format(alpha, fsize)
        )
        ifg = goldstein(ifg, alpha, fsize)
    rows, cols = ifg.shape
    geom = dict(res["geom"])
    geom.update(
        {
            "rows": rows,
            "cols": cols,
            "looks_range": looks[0],
            "looks_azimuth": looks[1],
            "azimuth_time_interval": ref.dt * looks[1],
            "range_pixel_spacing": geom.get("range_pixel_spacing", 0) * looks[0],
            "first_line_time": axis.iso(
                res["times"][0] + 0.5 * (looks[1] - 1) * ref.dt
            ),
        }
    )
    if looks != [1, 1]:
        geom["image_coordinates"] = (
            "x = (sample - (looks_range - 1) / 2) / looks_range + 0.5, "
            "y = rows - ((line - (looks_azimuth - 1) / 2) / looks_azimuth + 0.5)"
        )

    r_meta, s_meta = res["meta"], res["s_meta"]
    sw = ref.swath
    pol = sw.get("polarization")
    mission = r_meta["product"].get("mission") or sw.get("mission")
    pair_text = "{} / {}".format(
        r_meta["product"].get("product_name"), s_meta["product"].get("product_name")
    )
    t_ref = axis.datetime(ref.first_line_time)
    t_sec = axis.datetime(sec.first_line_time)
    stamp = "{}/{}".format(
        t_ref.strftime("%d %b %Y %H:%M:%S"), t_sec.strftime("%d %b %Y %H:%M:%S")
    )
    if t_sec < t_ref:
        stamp = "{}/{}".format(
            t_sec.strftime("%d %b %Y %H:%M:%S"), t_ref.strftime("%d %b %Y %H:%M:%S")
        )
    insar = {
        "reference": options["reference"],
        "secondary": options["secondary"],
        "reference_product": r_meta["product"].get("product_name"),
        "secondary_product": s_meta["product"].get("product_name"),
        "phase_removal": removal,
        "coherence_window": {"range": win[0], "azimuth": win[1]},
        "looks": {"range": looks[0], "azimuth": looks[1]},
        "goldstein": {"alpha": alpha, "fft_size": fsize, "window": FILTER_WINDOW}
        if alpha > 0
        else None,
        "baselines": info,
        "coregistration": s_meta.get("coregistration"),
        "wavelength": ref.wavelength,
        "sign_convention": "phase = arg(reference * conj(secondary)) - reference_phase",
    }
    secondary_meta = {
        "product": s_meta["product"],
        "swath": {k: v for k, v in s_meta["swath"].items() if k != "geolocation_grid"},
    }

    products = []
    for m in measures:
        if m == "phase":
            products.append(
                ("phase", np.angle(ifg), "radians", "wrapped interferometric phase")
            )
        elif m == "coherence" and coh is not None:
            products.append(("coherence", coh, "", "coherence"))
        elif m == "amplitude":
            products.append(("amplitude", np.abs(ifg), "", "interferogram amplitude"))
        elif m == "complex":
            products.append(("i", ifg.real, "", "interferogram real part"))
            products.append(("q", ifg.imag, "", "interferogram imaginary part"))
        elif m == "reference_phase":
            products.append(
                ("reference_phase", ref_phase, "radians", "removed synthetic phase")
            )
    env = gs.gisenv()
    mapset_dir = os.path.join(env["GISDBASE"], env["LOCATION_NAME"], env["MAPSET"])
    names = []
    for key, data, units, what in products:
        name = "{}_{}".format(stem, key)
        valid = np.isfinite(ifg) if key != "reference_phase" else np.isfinite(data)
        data = np.where(valid, data, np.nan)
        title = "{} {} {} {} {} ({})".format(
            mission, sw.get("mode"), sw.get("swath"), pol, what, pair_text
        )
        write_raster(name, data, title)
        gs.run_command(
            "r.support",
            map=name,
            title=title,
            units=units,
            source1=r_meta["product"].get("product_name") or "",
            source2=s_meta["product"].get("product_name") or "",
            description="i.sar.interferometry: {} removed; B_perp {:.1f} m; metadata in cell_misc/{}/description.json".format(
                removal, info["perpendicular"], name
            ),
            semantic_label="S1_{}_IFG_{}".format(pol, key.upper()),
            quiet=True,
        )
        gs.run_command("r.timestamp", map=name, date=stamp, quiet=True)
        gs.raster_history(name, overwrite=True)
        if key == "coherence":
            gs.run_command("r.colors", map=name, color="grey", quiet=True)
        elif key in ("phase", "reference_phase"):
            gs.run_command("r.colors", map=name, color="rainbow", quiet=True)
        meta = {
            "product": r_meta["product"],
            "swath": sw,
            "secondary": secondary_meta,
            "raster_geometry": geom,
            "interferometry": insar,
            "measure": key,
        }
        meta_dir = os.path.join(mapset_dir, "cell_misc", name)
        os.makedirs(meta_dir, exist_ok=True)
        with open(os.path.join(meta_dir, "description.json"), "w") as fd:
            json.dump(meta, fd, indent=1)
        names.append(name)
    gs.run_command("i.group", group=stem, input=names, quiet=True)
    npts = write_gcps(stem, ref, res["times"], rows, looks[1], looks[0], cols)
    gs.message(
        _("Imagery group <{}>: {} maps, {} ground control points").format(
            stem, len(names), npts
        )
    )


if __name__ == "__main__":
    sys.exit(main())
