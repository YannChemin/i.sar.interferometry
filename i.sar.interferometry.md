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

### Output

The maps are named `{output}_{measure}` (`{output}_bNN_{measure}` with
**-b**):

- `phase`: wrapped interferometric phase (radians), arg(m s\*) minus the
  synthetic phase;
- `coherence`: coherence magnitude (0 to 1);
- `amplitude`: interferogram amplitude |m s\*|;
- `i`, `q` (`complex`): real and imaginary parts of the interferogram;
- `reference_phase`: removed synthetic phase, wrapped.

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
coregistered with ESD avoid phase jumps at the burst seams. Phase
unwrapping, conversion to displacement or height and geocoding are left
to other tools.

The module requires NumPy.

## EXAMPLES

Interferogram and coherence of a pair coregistered with a DEM, the
topographic phase removed, with 4 × 1 looks and Goldstein filtering:

```sh
i.sar.coregistration reference=s1_20230112_iw2_vv secondary=s1_20230124_iw2_vv \
    output=s1_20230124_co dem=copernicus_dem_30m.tif extra=elevation
i.sar.interferometry reference=s1_20230112_iw2_vv secondary=s1_20230124_co \
    output=ifg_20230112_20230124 phase_removal=topography looks=4,1 filter_alpha=0.5
```

Geocode the phase and the coherence into a UTM project:

```sh
i.target group=ifg_20230112_20230124 location=sharjah_utm40n mapset=PERMANENT
i.rectify -t group=ifg_20230112_20230124 extension=_utm resolution=15
```

## REFERENCES

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
