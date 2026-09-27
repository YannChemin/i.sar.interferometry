# i.sar.interferometry

A [GRASS GIS](https://grass.osgeo.org/) addon that computes the
**interferogram, coherence, unwrapped phase and displacement
(subsidence) maps** of a Sentinel-1 TOPS SLC pair, in radar geometry and
geocoded, from a reference image imported by
[r.in.s1slc](https://github.com/YannChemin/r.in.s1slc) and a secondary
image coregistered on it by *i.sar.coregistration*.

```sh
r.in.s1slc -b input=<reference>.zip output=s1_20230112 swath=IW2 polarization=VV bbox=...
r.in.s1slc -b input=<secondary>.zip output=s1_20230124 swath=IW2 polarization=VV bbox=...
i.sar.coregistration reference=s1_20230112_iw2_vv secondary=s1_20230124_iw2_vv \
    output=s1_20230124_co dem=dem.tif extra=elevation
i.sar.interferometry reference=s1_20230112_iw2_vv secondary=s1_20230124_co \
    output=ifg phase_removal=topography looks=8,2 filter_alpha=0.5 \
    measure=coherence,vertical_displacement unwrap_mask=0.3 \
    reference_point=<lon>,<lat> target=<utm_project> dem=dem.tif
```

## Features

| Feature | Option | Notes |
|---|---|---|
| Synthetic phase | `phase_removal=none\|flat_earth\|topography` | exact orbit geometry; topography from the coregistration elevation maps |
| Products | `measure=phase,coherence,amplitude,complex,reference_phase` | FCELL maps in reference radar geometry |
| Coherence | `coherence_window=10,3` | after synthetic phase removal (SNAP coherence) |
| Debursting | default for burst inputs, `-b` to keep bursts | same geometry as the debursted reference |
| Multilook | `looks=range,azimuth` | complex averaging |
| Goldstein filter | `filter_alpha=`, `filter_size=` | SNAP algorithm, coherence unfiltered |
| Unwrapping | `measure=unwrapped_phase,...`, `unwrap_mask=`, `reference_point=` | minimum cost flow, SNAPHU-like statistical costs; C + OpenCL |
| Displacement | `measure=los_displacement,vertical_displacement` | subsidence map (vertical motion, negative down) |
| Geocoding | `target=`, `resolution=`, `dem=` | range-Doppler terrain correction into a projected project |
| Metadata | always | baselines, height of ambiguity, coregistration record, GCP group for *i.rectify* |

## Layout

| File | Role |
|---|---|
| `i.sar.interferometry.py` | GRASS module: interferogram, coherence, deburst, multilook, filter, displacement, geocoding |
| `sarunwrap_core.h` | per-pixel gradients, statistical costs and residues, compiled as C99 and as OpenCL C |
| `sarunwrap_kernels.cl` | OpenCL kernel entry point |
| `sarunwrap.c`, `sarunwrap.h` | `libsarunwrap`: device selection, minimum cost flow solver, integration, loaded through ctypes |

## Build and test

```sh
make MODULE_TOPDIR=$HOME/dev/grass
# Tests simulate TOPS pairs with the i.sar.coregistration test simulator
# and need the r.in.s1slc and i.sar.coregistration sources:
I_SAR_COREGISTRATION=$HOME/dev/i.sar.coregistration \
R_IN_S1SLC=$HOME/dev/r.in.s1slc/r.in.s1slc.py \
grass --tmp-project XY --exec python3 -m pytest tests
```

## License

GPL-2.0-or-later.
