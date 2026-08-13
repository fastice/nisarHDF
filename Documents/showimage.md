# showimage

Interactive viewer for NISAR HDF5 products, VRT, and GeoTIFF images. Displays 1–3 images side by side with a floating control palette for colormap selection, modulo display, and per-pane band switching. Images do not need to be the same size. Band data is loaded lazily — only read when first displayed.

## Entry points

| Command | Equivalent to |
|---|---|
| `showimage FILE [FILE ...]` | General viewer |
| `showvel FILE` | `showimage FILE --vel` — displays speed from a 2-band vx/vy file |
| `showoffsets FILE` | `showimage FILE --bands RangeOffsets AzimuthOffsets Correlation` |

## Usage

```
showimage [options] FILE [FILE ...]
```

**Positional:**

| Argument | Description |
|---|---|
| `FILE` | Input file(s) — up to 3; `.h5`/`.he5`/`.hdf5` for NISAR, `.vrt`/`.tif`/`.tiff` for GDAL, or a raw S1 `.pow` with a shared multilook geodat (see "Raw S1 power images") |

**Display options:**

| Flag | Default | Description |
|---|---|---|
| `--cmap CMAP` | `gray` | Colormap name (matplotlib) |
| `--vmin X` | 2nd percentile | Lower clip value |
| `--vmax X` | 98th percentile | Upper clip value |
| `--mod X` | off | Display image modulo X (useful for phase wrapping) |
| `--decFactor N` | auto-fit screen | Decimation factor (per-image when files differ in size) |
| `--fullRes` | — | Display at full resolution (equivalent to `--decFactor 1`) |
| `--right` | — | Place the control palette at the right edge of the screen, with the image and plot/profile windows opening to its left (mirrors the default left-to-right layout) — lets a second instance run without overlapping the first |
| `--mask MASKFILE` | — | External single-band mask file; pixels where the mask is 0 are treated as invalid. Handled exactly like a file's own embedded mask band — honored by default, toggleable via the **Mask** button (see "Mask toggles" below) — not baked in permanently |

**Velocity display (GDAL only):**

| Flag | Description |
|---|---|
| `--vel` | Read bands 1+2 as vx, vy for each file (1–3 files, side by side); display speed = √(vx²+vy²) mod 100 |
| `--log` | With `--vel`: log-scaled HSV speed rendering (1–3000 m/yr) |

**Band selection:**

| Flag | Description |
|---|---|
| `--bands BAND [BAND ...]` | Show 1–3 named bands from a single file (GDAL or NISAR HDF5) |

**GeoPackage overlay (GDAL only):**

| Flag | Default | Description |
|---|---|---|
| `--gpkg FILE` | — | Overlay Point/LineString/Polygon features from a GeoPackage (or any OGR vector source). The Legend window's dropdown lets you pick/switch which field colors the overlay without restarting. Not supported for NISAR HDF5 input |
| `--attribute FIELD` | first field | Initial attribute field to color features by (numeric → continuous colormap; text → fixed categorical palette). Requires `--gpkg`. Switch fields later from the Legend window |
| `--gpkgLayer NAME` | first layer | Layer name within `--gpkg` |
| `--gpkgCmap CMAP` | `viridis` | Colormap for a numeric `--attribute` |
| `--gpkgSize PX` | `4` | Point marker radius in pixels |
| `--gpkgFill` | off | Fill polygons (stippled) instead of outline-only |

**NISAR HDF5 options:**

| Flag | Default | Description |
|---|---|---|
| `--freq FREQ` | `frequencyA` | Frequency band to display |
| `--pol POL` | auto-detect | Polarization (e.g. `HH`, `VV`) |
| `--noCache` | — | Disable decimated-band cache; reduces memory use at the cost of re-reading from disk on each band switch |

## Raw S1 power images (`.pow`)

A raw headerless S1 power image named `P<scene>.<looks>.pow` (e.g. `P63811_669.10x2.pow`)
is displayed directly when a matching **shared multilook geodat** sits beside it —
`geodat<looks>.geojson` (preferred) or `geodat<looks>.in`, where `<looks>` is the token
just before `.pow` (`10x2` above), following the s1setup layout (`cloneSLCdir.py`). The
geodat supplies the grid dimensions (`nr` range × `na` azimuth, read via
`utilities.geodatrxa`); the `.pow` itself is read as **byte-swapped (big-endian) binary
float32**, stored azimuth-major.

The viewer shows the **amplitude = √power** (the square root compresses the dynamic range;
negative/border samples clip to 0). No map geotransform is applied — a `.pow` is in radar
(range/azimuth) geometry, so `--epsg`, `--gpkg`, and coordinate overlays don't apply. A
`.pow` files sometimes carry a few extra trailing azimuth rows beyond what the geodat
records; when the file is **longer** than the geodat's `na×nr`, the surplus is dropped (the
first `na×nr` samples are read) with a note on stderr. Only a file **shorter** than `na×nr`
is an error — usually a sign the wrong-`<looks>` geodat was picked up.

```
showimage P63811_669.10x2.pow          # amplitude, sized from geodat10x2.geojson/.in
```

## Multi-image display

When 2–3 files are given, each opens in its own pane. Images do not need to be the same size — each pane is independently decimated to fit its share of screen width, with its own scrollregion.

**Panel titles** show the filename when files differ, or just the band name when all panels come from the same file (e.g. `--bands`).

**Out-of-bounds clicks:** Pick mode reports the pixel value for each image independently. If a clicked position falls outside a smaller image, that pane shows `OOB` instead of a value. Col/Row plots similarly skip images where the column or row is out of range.

## NISAR HDF5 support

Pass one or more `.h5`/`.he5`/`.hdf5` NISAR product files directly. The viewer auto-detects the product type from the HDF5 path and lists all displayable fields as band-switch buttons in the palette.

### Supported products and fields

| Product | Fields displayed |
|---|---|
| **RIFG** | `wrappedInterferogram` (as phase), `coherenceMagnitude` |
| **RUNW** | `unwrappedPhase`, `coherenceMagnitude`, `connectedComponents`, `ionospherePhaseScreen`, `ionospherePhaseScreenUncertainty` |
| **ROFF** | per layer: `slantRangeOffset`, `alongTrackOffset`, `correlationSurfacePeak`, `snr`, `slantRangeOffsetVariance`, `alongTrackOffsetVariance`, `crossOffsetVariance` |
| **GUNW** | `unwrappedPhase`, `coherenceMagnitude`, `connectedComponents`, `ionospherePhaseScreen`, `ionospherePhaseScreenUncertainty` |
| **GOFF** | same per-layer fields as ROFF |
| **GCOV** | covariance terms (e.g. `HHHH`, `VVVV`) + `mask`, `numberOfLooks` |

Complex-valued fields (e.g. `wrappedInterferogram`) are displayed as phase (angle).

### Band caching

The decimated array for each band is cached in memory after the first display so that switching back to a previously viewed band is instant. Use `--noCache` to disable this on memory-constrained machines.

## GeoPackage overlay

`--gpkg FILE` overlays vector features from a GeoPackage (or any OGR vector source —
shapefile, etc.) directly on top of the displayed raster(s), colored by one attribute
field at a time. **The vector layer must already be in the same projected CRS as the
raster(s) being displayed — no reprojection is attempted**; world coordinates are mapped to
pixels using each raster's own geotransform.

- **Point** features are drawn as filled circles (`--gpkgSize` radius, black outline).
- **LineString** features are drawn as lines.
- **Polygon** features are drawn outline-only by default (so the underlying raster stays
  visible); pass `--gpkgFill` to fill them (stippled).
- A numeric attribute gets a continuous colormap (`--gpkgCmap`, default `viridis`); a
  text/categorical attribute gets a fixed 10-color qualitative palette, assigned in
  first-seen order. Missing/unparseable values are drawn mid-gray in either case.
- A separate **Legend** window opens automatically, positioned just outside the image
  window on the side opposite the Controls palette (right by default; left with `--right`),
  top-aligned with it. It shows a colorbar (numeric) or a swatch + label list (categorical)
  for the current attribute. If the layer has more than one field, a **Color by:** dropdown
  appears above the legend — picking a different field recolors the overlay (autoscaled)
  and rebuilds the legend in place, no restart needed. `--attribute` just sets which field
  is selected initially (default: the first field in the layer).
- For a **numeric** attribute, the Legend also shows **Min**/**Max** entries — pre-filled
  with the autoscaled range (the data's own min/max) — plus **Apply** (recolor using the
  entered range; values outside it clamp to the colormap's end colors) and **Auto** (revert
  to the autoscaled range and refresh the entries to match). Switching fields via the
  dropdown always resets to that field's own autoscaled range.
- A **GPKG** button appears in the control palette to toggle the overlay's visibility
  without closing the legend.
- With multiple panes (2–3 files, or a `--diff`/`--add` combined pane), the same overlay is
  drawn on every pane that has its own georeferencing — each pane converts world coordinates
  to its own pixel space independently, so this works even if the panes differ in
  decimation factor or size.
- Not supported for NISAR HDF5 input (no comparable geotransform is available there).

## Mask toggles

Two independent sources of masking feed into the same toggleable mechanism:

1. A file's own **embedded GDAL mask band** — the same "embedded VRT dataset mask band"
   concept the GIT64 C binaries' `-noMask` flag refers to (a shared per-dataset mask, a
   per-band mask, or an alpha band; *not* just a NoData value, which `showimage` already
   handles separately). Detected automatically, no flag needed.
2. `--mask MASKFILE` — an external single-band mask file (pixels where it's 0 are invalid).

Both are combined (logical AND — a pixel is valid only if *neither* source marks it
invalid) into one effective mask per pane, honored by default:

- **Mask** button — toggles honoring the combined mask on/off; when off, the raw,
  unmasked data is shown instead. Applies to every pane/band that has a mask (checked
  independently per pane and, for multi-band files, re-checked on every band switch —
  a per-band mask can differ from band to band, though most files share one for the
  whole dataset). Only shown if a mask is actually detected in the file(s) as initially
  loaded.
- **InvMask** button — flips which sense counts as valid for the *effective combined*
  mask, regardless of source: an embedded VRT mask band alone, `--mask` alone, or both
  together. This replaces the old separate `-invMask` flag with a live toggle that works
  no matter where the mask came from (applied at display time, not baked into any one
  source). Same visibility condition as **Mask** — shown whenever there's a mask to
  invert, not just when `--mask` was given.
- Neither is shown for NISAR HDF5 input (no `--mask` support there, and no embedded mask
  band concept).

## Interactive controls

The floating palette (left window) provides:

- **Pick mode** — click a pixel to read its value in all panes simultaneously
- **Profile mode** — click two points to extract and plot a line profile
- **Col Plot / Row Plot** — click a pixel to plot that column or row across all panes
- **Lines** — toggle visibility of all profile/plot overlay lines
- **GPKG** *(only shown with `--gpkg`)* — toggle visibility of the GeoPackage overlay
- **Mask** *(only shown if a mask is detected)* — toggle honoring the combined mask
  (embedded VRT mask band and/or `--mask`); applied by default (see "Mask toggles" above)
- **InvMask** *(only shown if a mask is detected)* — flip which sense counts as valid for
  the effective combined mask, from whichever source(s) it came from
- **Sync** *(multi-image only)* — toggle synchronized scrolling across panes; defaults to on when all images are the same size, off when they differ
- **Colormap selector** — live colormap switching applied to all panes
- **Modulo input** — enter a value and press Return to apply modulo display per pane
- **Band buttons** — one button per available band/field; click to switch that pane

### Plot window controls

The Col Plot, Row Plot, and Profile windows each have an axis control row at the bottom:

| Control | Description |
|---|---|
| `Y: [min] to [max]` | Set Y-axis limits (leave blank to leave that limit unchanged) |
| `X: [min] to [max]` | Set X-axis limits (Col/Row plots only) |
| **Apply** | Apply the entered limits to all subplots |
| **Auto** | Reset both axes to auto-scale and populate the entry fields with the resulting limits |
| **Log Y** | Toggle logarithmic Y axis (shows ✓ when active) |

Col/Row plot windows also have **Single** (overlay all panes in one subplot), **Save** (PNG/PDF/SVG), and **Clear** (remove plotted lines).

## Examples

```bash
# Display a RIFG interferogram (wrapped phase; coherence button in palette)
showimage NISAR_L1_PR_RIFG_*.h5

# Display with a custom colormap and phase limits
showimage NISAR_L1_PR_RIFG_*.h5 --cmap seismic --vmin -3.14 --vmax 3.14

# Compare two NISAR products (different sizes are allowed; scroll sync defaults to off)
showimage NISAR_L1_PR_RIFG_*.h5 NISAR_L1_PR_RUNW_*.h5

# Display specific bands from a NISAR ROFF product
showimage NISAR_L1_PR_ROFF_*.h5 --bands layer1/slantRangeOffset layer1/alongTrackOffset

# Display a GOFF geocoded offset product, frequencyB
showimage NISAR_L2_PR_GOFF_*.h5 --freq frequencyB

# Velocity file (VRT with vx, vy bands)
showvel velocity.vrt

# Offset correlation quicklook
showoffsets offsets.vrt

# Overlay a QC GeoPackage of tie points, colored by residual sigma
showimage velocity.tif --gpkg tiepoints.gpkg --attribute sigmaRBaseline

# Overlay frame footprint polygons, colored by a categorical field
showimage mosaic.tif --gpkg frames.gpkg --attribute direction --gpkgFill
```
