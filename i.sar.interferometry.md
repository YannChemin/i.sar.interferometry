## DESCRIPTION

*i.sar.interferometry* forms the interferogram of a Sentinel-1 TOPS SLC
pair and derives its phase, coherence and amplitude. The **reference**
image is a complex image imported by *r.in.s1slc*; the **secondary**
image is the output of *i.sar.coregistration* for that reference. The
processing follows ESA SNAP (microwave toolbox) *Interferogram*,
*TOPSAR-Deburst*, *Multilook* and *Goldstein Phase Filtering*.

- **Interferogram**: m s<sup>\*</sup> e<sup>-jφ<sub>ref</sub></sup>,
  m and s being the reference and coregistered secondary complex values
  and φ<sub>ref</sub> the synthetic phase selected by
  **phase_removal**: nothing, the flat-earth phase of the WGS84
  ellipsoid, or the phase of the terrain given by the `elevation` maps
  that *i.sar.coregistration* writes with **extra=elevation** and a DEM.
  The synthetic phase is -4π (R<sub>ref</sub> - R<sub>sec</sub>) / λ,
  the slant ranges of each ground point from the reference and secondary
  orbits; it is computed exactly on a grid of nodes (every 10 lines and
  100 samples, at two heights enclosing the terrain) and interpolated.
- **Coherence**: |⟨m s<sup>\*</sup>e<sup>-jφ<sub>ref</sub></sup>⟩| /
  √(⟨|m|²⟩⟨|s|²⟩) over a moving window of **coherence_window** samples
  and lines, after removal of the synthetic phase (whose fringes would
  otherwise bias the estimate low).
- **Debursting**: with burst maps (*r.in.s1slc* **-b**), the burst
  interferograms are merged like *r.in.s1slc* debursts images (SNAP
  *TOPSAR-Deburst*): rows are sampled every azimuth time interval, the
  burst switching in the middle of each overlap. The output then has
  exactly the geometry of the debursted reference image. The **-b** flag
  keeps one set of maps per burst instead.
- **Multilooking**: **looks** averages the complex interferogram and the
  coherence over blocks of samples and lines.
- **Goldstein filtering**: with **filter_alpha** above 0 the
  (multilooked) interferogram is filtered in blocks of **filter_size**
  samples overlapping by three quarters: the spectrum of each block is
  weighted by its 3 × 3 smoothed magnitude to the power alpha, and the
  blocks are recombined with triangular weights. The coherence is not
  filtered.
- **Phase unwrapping** (products `unwrapped_phase`, `los_displacement`,
  `vertical_displacement`, `unwrap_component`): minimum cost flow
  unwrapping with statistical costs, in the spirit of SNAPHU (see below),
  of the (multilooked, filtered) phase.
- **Displacement**: the line of sight displacement is
  -λ φ<sub>unw</sub> / 4π, positive towards the satellite; the vertical
  displacement is the LOS displacement divided by the cosine of the
  incidence angle, assuming a vertical motion only: subsidence is
  negative.
- **Geocoding** (**target**): range-Doppler terrain correction of every
  product into a projected (or geographic) project.

### Phase unwrapping

The residues of the wrapped phase (loops of 2 × 2 pixels whose wrapped
gradients do not sum to zero) are joined to each other or to the image
border by a minimum cost flow in the network of the loops (Costantini,
1998): the flow across a pixel edge is the number of cycles added to the
wrapped gradient there. The cost of adding n cycles to a gradient Δ is
statistical, as in SNAPHU: ((Δ + 2πn)² - Δ²) / 2σ², σ² = (1 - γ²) /
(2Lγ²) being the phase variance of L looks (the product of **looks**) at
the coherence γ of the edge. Edges of coherent pixels are expensive to
cut, decorrelated ones cheap, and edges touching null pixels free. The
flow is solved exactly by successive shortest paths with node potentials
and unit augmentations, and the corrected gradients are integrated over
each connected component of non-null pixels.

Pixels whose coherence is below **unwrap_mask** are not unwrapped. They
are null in the unwrapped products and separate connected components
(`unwrap_component`, 1 for the largest): the phase offset between two
components is unknown. Each component is referenced separately: the
component of the **reference_point** (a stable longitude, latitude) to
its value there, the others to their median; without reference point
every component has a zero median.

The per-pixel preparation (wrapped gradients, costs, residues) runs on an
OpenCL device or on the host with OpenMP (**device**, **platform**,
**nprocs**); the network flow and the integration run on the host CPU.

### Geocoding

With **target**, every product is also written, with the same name, in
the **target_mapset** of that project (in the same GISDBASE). Each target
cell centre is located on the **dem** (reprojected to the target grid,
EGM96 heights converted with **dem_height=geoid**), or at the mean
terrain height of the annotation without DEM, and projected into the
reference radar geometry by its zero-Doppler time and slant range; the
product is sampled there, bilinearly (nearest neighbour for the wrapped
phases and the components). The radar positions are computed exactly on
nodes every 16 cells for two heights enclosing the terrain and
interpolated. The target grid covers the product footprint at
**resolution** (default: the ground spacing of the product pixels).

### Output

The maps are named `{output}_{measure}` (`{output}_bNN_{measure}` with
**-b**):

- `phase`: wrapped interferometric phase (radians), arg(m s\*) minus the
  synthetic phase;
- `coherence`: coherence magnitude (0 to 1);
- `amplitude`: interferogram amplitude |m s\*|;
- `i`, `q` (`complex`): real and imaginary parts of the interferogram;
- `reference_phase`: removed synthetic phase, wrapped;
- `incidence_angle`: incidence angle on the ellipsoid (degrees);
- `unwrapped_phase`: unwrapped phase (radians), referenced as above;
- `los_displacement`: line of sight displacement (m), positive towards
  the satellite;
- `vertical_displacement`: vertical displacement (m), negative for
  subsidence;
- `unwrap_component`: connected components of the unwrapping.

They are FCELL maps in the radar geometry of the reference (one cell per
sample and line, or per look), with timestamps spanning the two
acquisitions, semantic labels (e.g. `S1_VV_IFG_PHASE`) and a
`cell_misc/<map>/description.json` file holding the reference product and
annotation, the secondary product and annotation, the raster geometry
(debursting segments, looks) and an `interferometry` section: phase
removal, windows, filter, the coregistration record and the baselines
at the scene centre (perpendicular baseline, positive when the secondary
orbit is above the reference one; parallel baseline; temporal baseline;
incidence angle; height of ambiguity).

An imagery group `{output}` holds the maps and ground control points
from the reference geolocation grid, for geocoding with *i.rectify*.

## NOTES

The current project must be unprojected (XY). The computational region
is not used.

The interferogram is only as good as the coregistration: burst maps
coregistered with ESD avoid phase jumps at the burst seams.

The displacement of a single interferogram includes the atmospheric
delay difference of the two acquisitions and any residual orbit or DEM
error; the vertical displacement also ignores horizontal motion. The
unwrapping holds the whole image in memory (about 60 bytes per pixel):
multilook large images first. Layover and shadow are not masked by the
geocoding.

The module requires NumPy, and the GDAL Python bindings for the
geocoding. The unwrapping library `libsarunwrap` needs a C compiler with
OpenMP and the OpenCL headers and ICD loader.

## EXAMPLES

Interferogram and coherence of a pair coregistered with a DEM, the
topographic phase removed, with 4 × 1 looks and Goldstein filtering:

```sh
i.sar.coregistration reference=s1_20230112_iw2_vv secondary=s1_20230124_iw2_vv \
    output=s1_20230124_co dem=copernicus_dem_30m.tif extra=elevation
i.sar.interferometry reference=s1_20230112_iw2_vv secondary=s1_20230124_co \
    output=ifg_20230112_20230124 phase_removal=topography looks=4,1 filter_alpha=0.5
```

Subsidence map: unwrapped vertical displacement, referenced to a stable
point, geocoded with the DEM into a UTM project at 20 m:

```sh
grass -c EPSG:32640 $HOME/grassdata/sharjah_utm40n -e
i.sar.interferometry reference=s1_20230112_iw2_vv secondary=s1_20230124_co \
    output=defo_20230112_20230124 phase_removal=topography looks=8,2 filter_alpha=0.5 \
    measure=coherence,vertical_displacement,unwrap_component unwrap_mask=0.3 \
    reference_point=55.402,25.318 target=sharjah_utm40n resolution=20 \
    dem=copernicus_dem_30m.tif
```

The radar geometry products can also be geocoded with their ground
control points:

```sh
i.target group=ifg_20230112_20230124 location=sharjah_utm40n mapset=PERMANENT
i.rectify -t group=ifg_20230112_20230124 extension=_utm resolution=15
```

## REFERENCES

- M. Costantini, *A novel phase unwrapping method based on network
  programming*, IEEE Transactions on Geoscience and Remote Sensing,
  36(3), 813-821, 1998.
- C. W. Chen and H. A. Zebker, *Two-dimensional phase unwrapping with
  use of statistical models for cost functions in nonlinear optimization*,
  Journal of the Optical Society of America A, 18(2), 338-351, 2001
  (SNAPHU).
- R. M. Goldstein and C. L. Werner, *Radar interferogram filtering for
  geophysical applications*, Geophysical Research Letters, 25(21),
  4035-4038, 1998.
- N. Yagüe-Martínez et al., *Interferometric Processing of Sentinel-1
  TOPS Data*, IEEE Transactions on Geoscience and Remote Sensing, 54(4),
  2220-2234, 2016.
- ESA SNAP microwave toolbox:
  <https://github.com/senbox-org/microwave-toolbox>

## SEE ALSO

*[i.group](https://grass.osgeo.org/grass-stable/manuals/i.group.html),
[i.rectify](https://grass.osgeo.org/grass-stable/manuals/i.rectify.html),
[i.sar.coregistration](i.sar.coregistration.html),
[r.in.s1slc](r.in.s1slc.html)*

## AUTHORS

Yann Chemin
