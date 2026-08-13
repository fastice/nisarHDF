# CLAUDE.md — nisarhdf

Python library for reading NISAR Level-1 and Level-2 HDF5 products and exporting data to GrIMP binary flat-file format. Used heavily by `nisargrimpworkflow`. See the [packages CLAUDE.md](../CLAUDE.md) for pipeline context.

## Class Hierarchy

```
nisarBaseHDF
├── nisarBaseRangeDopplerHDF      (range-Doppler gridded products)
│   ├── nisarRSLCHDF              RSLC — single-look complex
│   ├── nisarRIFGHDF              RIFG — interferogram
│   ├── nisarRUNWHDF              RUNW — unwrapped interferogram  ← key for InSAR
│   ├── nisarROFFHDF              ROFF — range/azimuth offsets    ← key for speckle
│   └── nisarGOFFHDF              GOFF — geocoded offsets
└── nisarBaseGeocodedHDF          (geocoded products)
    ├── nisarGCOVHDF              GCOV — geocoded covariance
    └── nisarGUNWHDF              GUNW — geocoded unwrapped phase
```

## Usage Pattern

```python
import nisarhdf

myROFF = nisarhdf.nisarROFFHDF()
myROFF.openHDF('/path/to/NISAR_L1_PR_ROFF_*.h5')

# Access data as attributes (numpy arrays or scalars)
rg_offsets = myROFF.slantRangeOffset     # 2D float32 array
az_offsets = myROFF.alongTrackOffset     # 2D float32 array
corr = myROFF.correlationSurfacePeak     # 2D float32 array
nr = myROFF.OffsetRangeSize              # int
na = myROFF.OffsetAzimuthSize            # int
epsg = myROFF.epsg                       # int (3413=Greenland, 3031=Antarctica)
date = myROFF.Date                       # datetime
bw = myROFF.rangeBandwidth               # float (Hz)

# Write to GrIMP binary flat files
myROFF.writeData('/out/dir/NISARoffsets',
                 bands=['slantRangeOffset', 'alongTrackOffset',
                        'correlationSurfacePeak'],
                 tiff=False,
                 byteOrder='MSB',    # big-endian; matches GIT64 expectation
                 grimp=True,
                 saveMatch=True,
                 scaleToPixels=True)

# Write .dat metadata file for use by C programs
myROFF.writeOffsetsDatFile('/out/dir/offsets.dat', geodat1='geodat3x6.geojson')

# Discard outliers
myROFF.removeOutlierOffsets('correlationSurfacePeak',
                             thresholds=[0.07, 0.05, 0.025])

# Apply mask
myROFF.applyMask('/out/dir/workingDir/offsets.mask.vrt')

# Get GDAL GeoTransform (for VRT construction)
gt = myROFF.getGeoTransform(grimp=True, tiff=False)
```

## Key Attributes (common across product classes)

| Attribute | Type | Description |
|---|---|---|
| `Date` | `datetime` | Reference acquisition date |
| `secondaryDate` | `datetime` | Secondary (for interferometric pairs) |
| `referenceOrbit` | `int` | Reference orbit number |
| `secondaryOrbit` | `int` | Secondary orbit number |
| `epsg` | `int` | Projection EPSG code |
| `NumberRangeLooks` | `int` | Multi-look factor in range |
| `NumberAzimuthLooks` | `int` | Multi-look factor in azimuth |
| `rangeBandwidth` | `float` | Range bandwidth in Hz |
| `r0`, `a0` | `float` | Near-range slant range, first azimuth time |
| `deltaR`, `deltaA` | `float` | Range and azimuth sample spacing |

## Key Methods

| Method | Description |
|---|---|
| `openHDF(path)` | Open the HDF5 file; populates all attributes |
| `writeData(path, bands, ...)` | Write bands to GrIMP binary flat files |
| `writeOffsetsDatFile(path, geodat1)` | Write `.dat` sidecar for C programs |
| `removeOutlierOffsets(field, thresholds)` | Mask outliers by correlation |
| `applyMask(vrtFile)` | Apply binary mask (0=invalid) to offset arrays |
| `getGeoTransform(grimp, tiff)` | Return GDAL geotransform for VRT creation |
| `getRangeBandWidth()` | Populate `rangeBandwidth` attribute |

## Utility Functions

```python
nisarhdf.readVrtAsXarray(vrtFile)     # Read a VRT as xarray Dataset
nisarhdf.formatGeojson(string)        # Pretty-print a GeoJSON string
nisarhdf.writeMultiBandVrt(...)       # Write a multi-band VRT
```

## GrIMP Output Format

When `writeData(..., grimp=True, tiff=False)` is called:
- Each band is written as a separate file: `{path}.layer{N}.dr`, `.da`, `.sr`, `.sa`
- Format: big-endian float32 (`>f4`) when `byteOrder='MSB'`
- `scaleToPixels=True` converts offsets from metres to pixel units
- `saveMatch=True` also saves the correlation peak as a separate file

The `.dat` file written by `writeOffsetsDatFile` follows the GrIMP metadata format: `#BEGINDATA` header, key-value pairs, `&` terminator. Read by `getDataString()` in GIT64's `clib/standard.c`.

## `geolocationGrid` cube methods (squint / geometry diagnostics)

`nisarBaseHDF.py` has `losUnitVectorCube()`, `alongTrackUnitVectorCube()`, and
`elevationAngleCube()` (using `interpGrid()`/`setupDataCube()`) for proper interpolation of the
RUNW `science/LSAR/RUNW/metadata/geolocationGrid` cube (`losUnitVectorX/Y`,
`alongTrackUnitVectorX/Y`, `elevationAngle`, etc. — official L1 processor output, NISAR ATBD
JPL D-95677 §3.4–3.8) at arbitrary (x,y,z). `computeAngles()`'s own docstring notes its
zero-Doppler-plane elevation-angle calculation is the *idealized* counterpart to the dataCube's
elevation angle, which "includes the squint."

These cube methods are the correct way to extract real squint (the deviation of `losUnitVector`
from broadside relative to `alongTrackUnitVector`) for any future work — the `nisarErrors`
package's `plotSquintError.py` (which found and quantified this effect for GrIMP's `mosaic3d`;
see `~/PycharmProjects/packages/nisarErrors/Documents/plotSquintError.md` and
`~/progs/GIT64/mosaicSource/CLAUDE.md` "Squint (residual Doppler) sensitivity") currently uses a
cruder nearest-grid-point lookup instead of these methods — a known limitation of that analysis,
not of this package.

**Implemented (stage 1 of 3 — extraction; merge in `SetupNISAR` and the `mosaic3d` correction are
the other two stages, all now also implemented — see `nisargrimpworkflow/CLAUDE.md` and
`~/progs/GIT64/mosaicSource/CLAUDE.md`):** `nisarBaseHDF.getSquintAnglePolynomial()`
samples a 9×9 range/azimuth grid via `losUnitVectorCube`/`alongTrackUnitVectorCube` (height fixed
at the cube's own mid-level `heightAboveEllipsoid`, negligible height-sensitivity per the
analysis doc) and fits `squint(r,a) = c0 + c1 r' + c2 a' + c3 a'^2 + c4 r' a' + c5 r'^2` via
`np.linalg.lstsq`, where `r' = r - MLCenterRange`, `a' = a - MLMidZeroDopplerTime` (centering on
the frame's *existing* attributes, not a freshly-computed mean — needed because raw slant range
(~1e6 m) / raw zero-Doppler time (~1e4 s of day) are poorly conditioned for a quadratic fit).
Unlike `measureSquint()`, this includes the look-direction sign on the broadside term:
`squint = wrap180(heading(los) - heading(track)) - 90°·L`, `L = +1` left-looking, `-1` right.
Verified against a real Greenland RUNW (`track-58`, left-looking): fitted `c0` = 1.65824°
vs. `measureSquint()`'s 1.65808° (expected near-exact match since `L=+1` there); fit residual std
0.00087° over a 76 km / 36 s frame spanning squint 1.539°–1.826° — well inside the ~0.01°
noise floor `plotSquintError.md` documents.

`nisarRUNWHDF.getSquint(secondary=False)` overrides the base stub and is the only caller: it sets
`self.squintAnglePolynomial = {'coefficients': [...], 'refRange': ..., 'refAzimuthTime': ...}`,
written into `geodatMxN.geojson` by the existing `genGeodatProperties()`/`writeGeodatGeojson()`.
**The legacy scalar `Squint` field is left untouched** (still `0.0` from the base-class stub) —
older code uses it for a deskew time delay to zero Doppler, a different quantity from this angle
fit, and must not be repurposed even though the names collide. **Reference image only** —
`self.secondary.h5 = self.h5` means the secondary shares the reference's `geolocationGrid` cube,
whose zero-Doppler time axis only spans the reference pass's absolute time; sampling it at the
secondary's own (different repeat-pass day) azimuth times would land outside the cube and
silently return NaN. So the secondary keeps `squintAnglePolynomial=None` —
`getSquint(secondary=True)` returns right after the base-class defaults, skipping the fit.

**Flat fields, added for stage 3's C-side consumer:** `mosaic3d` reads geodats via GDAL's OGR
GeoJSON driver, which can't read a nested object like `squintAnglePolynomial` — only flat
scalars/lists (the existing `SV_Pos_N`/`CenterLatLon` precedent). `genGeodatProperties()`
additively writes three top-level keys mirroring the nested dict — `squintCoefficients` (the same
6-element list), `squintRefRange`, `squintRefAzimuthTime` (all `None` when
`squintAnglePolynomial` is `None`) — alongside it, not instead of it; nothing here changed for
Python-side consumers.

## Notes

- `nisarfunc` is an older, now-deprecated package that preceded `nisarhdf`. It has a similar but less complete `nisarVel.py` and `nisarBase2D.py`. Do not add new code there; use `nisarhdf` instead.
- `nisardev` builds on top of `nisarhdf` and adds higher-level notebook-oriented classes (`nisarImage`, `nisarVel`, `nisarVelSeries`).
