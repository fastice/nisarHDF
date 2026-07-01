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
| `FILE` | Input file(s) — up to 3; `.h5`/`.he5`/`.hdf5` for NISAR, `.vrt`/`.tif`/`.tiff` for GDAL |

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

**Velocity display (GDAL only):**

| Flag | Description |
|---|---|
| `--vel` | Read bands 1+2 as vx, vy; display speed = √(vx²+vy²) mod 100 |
| `--log` | With `--vel`: log-scaled HSV speed rendering (1–3000 m/yr) |

**Band selection:**

| Flag | Description |
|---|---|
| `--bands BAND [BAND ...]` | Show 1–3 named bands from a single file (GDAL or NISAR HDF5) |

**NISAR HDF5 options:**

| Flag | Default | Description |
|---|---|---|
| `--freq FREQ` | `frequencyA` | Frequency band to display |
| `--pol POL` | auto-detect | Polarization (e.g. `HH`, `VV`) |
| `--noCache` | — | Disable decimated-band cache; reduces memory use at the cost of re-reading from disk on each band switch |

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

## Interactive controls

The floating palette (left window) provides:

- **Pick mode** — click a pixel to read its value in all panes simultaneously
- **Profile mode** — click two points to extract and plot a line profile
- **Col Plot / Row Plot** — click a pixel to plot that column or row across all panes
- **Lines** — toggle visibility of all profile/plot overlay lines
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
```
