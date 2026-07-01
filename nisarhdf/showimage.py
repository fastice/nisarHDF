#!/usr/bin/env python3
import argparse
import math
import os
import sys
import numpy as np

try:
    from osgeo import gdal
except ImportError:
    gdal = None

DPI = 100
CBAR_PX = 120   # pixels reserved for colorbar panel
SCROLLBAR_W = 18  # scrollbar widget thickness
LABEL_H = 22      # per-image title label height
QUIT_H = 34       # quit button row height
DECO_H = 95       # WM title bar (~37) + buffer; task bar already excluded by wm maxsize
CMAPS = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'hot', 'coolwarm',
         'RdBu', 'seismic', 'bwr', 'jet', 'rainbow', 'turbo', 'hsv']


def getScreenSize():
    """Return (width_px, height_px) of the usable desktop (excluding taskbars).

    Uses 'wm maxsize' which is the maximum window size the WM will allow,
    i.e. the work area after subtracting panels and taskbars.  Falls back
    to raw screen dimensions if the query fails.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        try:
            result = root.eval('wm maxsize .')
            w, h = (int(x) for x in result.split())
        except Exception:
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def blockAverage(arr, factor):
    """Block-average arr in factor×factor tiles using nanmean."""
    ny, nx = arr.shape[:2]
    ny2 = (ny // factor) * factor
    nx2 = (nx // factor) * factor
    a = arr[:ny2, :nx2]
    with np.errstate(all='ignore'):
        if a.ndim == 2:
            return np.nanmean(
                a.reshape(ny2 // factor, factor, nx2 // factor, factor),
                axis=(1, 3))
        nb = a.shape[2]
        return np.nanmean(
            a.reshape(ny2 // factor, factor, nx2 // factor, factor, nb),
            axis=(1, 3))


def readBand(ds, b):
    band = ds.GetRasterBand(b)
    data = band.ReadAsArray().astype(np.float32)
    nd = band.GetNoDataValue()
    if nd is None:
        nd = -2e9
    data[data == np.float32(nd)] = np.nan
    return data


def getBandNames(ds):
    """Return list of band name strings (one per band), using description or metadata."""
    names = []
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b)
        name = (band.GetDescription()
                or band.GetMetadata().get('Description')
                or f'Band{b}')
        names.append(name)
    return names


def findBandByName(ds, name):
    """Return 1-based band number whose description matches name, or None.

    Checks both GetDescription() and the 'Description' metadata item,
    since different writers use different conventions.
    """
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b)
        if band.GetDescription() == name:
            return b
        if band.GetMetadata().get('Description') == name:
            return b
    return None


def isNisarOriginLowerProduct(filename):
    """True if filename looks like a NISAR RIFG/RUNW/ROFF-derived product (native
    range/azimuth grid, origin-lower convention) rather than an unrelated file that
    happens to start with the letter 'R' (e.g. a RACMO SMB correction grid)."""
    base = os.path.basename(filename)
    return any(p in base for p in ('RIFG', 'RUNW', 'ROFF'))


# ---------------------------------------------------------------------------
# NISAR HDF5 support
# ---------------------------------------------------------------------------

_NISAR_KNOWN_PRODUCTS = ['RIFG', 'RUNW', 'ROFF', 'GUNW', 'GOFF', 'GCOV']


def openNisarH5(filepath, frequency='frequencyA', pol=None):
    """Open a NISAR HDF5 file and return lazy band loaders.

    Returns (nx, ny, product_type, loaders, h5) where:
      loaders  — dict {band_name: callable → float32 ndarray}
      h5       — open h5py.File; caller must keep open until display is done
    """
    try:
        import h5py
    except ImportError:
        sys.exit('h5py not available — install h5py')

    h5 = h5py.File(filepath, 'r')
    try:
        lsar = h5['science']['LSAR']
    except KeyError:
        h5.close()
        sys.exit(f'showimage: {filepath} does not look like a NISAR HDF5 file')

    product = None
    for p in _NISAR_KNOWN_PRODUCTS:
        if p in lsar:
            product = p
            break
    if product is None:
        h5.close()
        sys.exit(f'showimage: unrecognised NISAR product type in {filepath} '
                 f'(found: {list(lsar.keys())})')

    prod_grp = lsar[product]
    bands_key = 'swaths' if product in ('RIFG', 'RUNW', 'ROFF') else 'grids'
    freq_grp = prod_grp[bands_key][frequency]

    # Detect polarization (not needed for GCOV which uses covariance terms)
    available_pol = pol
    if product != 'GCOV' and available_pol is None:
        for pol_key in ('listOfPolarizations',):
            if pol_key in freq_grp:
                try:
                    pols = [p.decode() if isinstance(p, bytes) else str(p)
                            for p in freq_grp[pol_key]]
                    if pols:
                        available_pol = pols[0]
                except Exception:
                    pass
                break
        if available_pol is None:
            # Probe group keys in the relevant subgroup
            probe_grp = None
            if product in ('RIFG', 'RUNW'):
                probe_grp = freq_grp.get('interferogram')
            elif product == 'ROFF':
                probe_grp = freq_grp.get('pixelOffsets')
            elif product == 'GUNW':
                probe_grp = (freq_grp.get('unwrappedInterferogram')
                             or freq_grp.get('wrappedInterferogram'))
            elif product == 'GOFF':
                probe_grp = freq_grp.get('pixelOffsets')
            if probe_grp is not None:
                for p in ('HH', 'VV', 'HV', 'VH'):
                    if p in probe_grp:
                        available_pol = p
                        break
        if available_pol is None:
            h5.close()
            sys.exit(f'showimage: no supported polarization found in {filepath}')

    loaders = {}
    ny = nx = None

    def _make_loader(ds, is_phase=False):
        def load():
            data = ds[:]
            fv = ds.fillvalue
            if np.iscomplexobj(data):
                # Detect fill locations before conversion; fv may be complex
                fill_mask = (data == fv) if fv is not None else None
                arr = np.angle(data).astype(np.float32) if is_phase \
                    else np.abs(data).astype(np.float32)
                if fill_mask is not None:
                    arr[fill_mask] = np.nan
            else:
                arr = data.astype(np.float32)
                if fv is not None:
                    try:
                        if not np.isnan(float(fv)):
                            arr[arr == np.float32(fv)] = np.nan
                    except (TypeError, ValueError):
                        pass
            return arr
        return load

    def _register(name, ds, is_phase=False):
        nonlocal ny, nx
        loaders[name] = _make_loader(ds, is_phase=is_phase)
        if ny is None and hasattr(ds, 'shape') and len(ds.shape) >= 2:
            ny, nx = ds.shape[:2]

    if product in ('RIFG', 'RUNW'):
        intf_grp = freq_grp['interferogram'][available_pol]
        field_is_phase = {
            'wrappedInterferogram': True,
            'coherenceMagnitude': False,
            'unwrappedPhase': False,
            'connectedComponents': False,
            'ionospherePhaseScreen': False,
            'ionospherePhaseScreenUncertainty': False,
        }
        for fname, is_phase in field_is_phase.items():
            if fname in intf_grp:
                _register(fname, intf_grp[fname], is_phase=is_phase)

    elif product == 'ROFF':
        off_grp = freq_grp['pixelOffsets'][available_pol]
        layers = sorted(k for k in off_grp.keys() if k.startswith('layer'))
        layer_fields = ['slantRangeOffset', 'alongTrackOffset', 'correlationSurfacePeak',
                        'snr', 'slantRangeOffsetVariance', 'alongTrackOffsetVariance',
                        'crossOffsetVariance']
        for layer in layers:
            for fname in layer_fields:
                if fname in off_grp[layer]:
                    _register(f'{layer}/{fname}', off_grp[layer][fname])

    elif product == 'GUNW':
        pt_key = ('unwrappedInterferogram' if 'unwrappedInterferogram' in freq_grp
                  else 'wrappedInterferogram')
        intf_grp = freq_grp[pt_key][available_pol]
        field_is_phase = {
            'unwrappedPhase': False,
            'coherenceMagnitude': False,
            'connectedComponents': False,
            'ionospherePhaseScreen': False,
            'ionospherePhaseScreenUncertainty': False,
            'wrappedInterferogram': True,
        }
        for fname, is_phase in field_is_phase.items():
            if fname in intf_grp:
                _register(fname, intf_grp[fname], is_phase=is_phase)

    elif product == 'GOFF':
        off_grp = freq_grp['pixelOffsets'][available_pol]
        layers = sorted(k for k in off_grp.keys() if k.startswith('layer'))
        layer_fields = ['slantRangeOffset', 'alongTrackOffset', 'correlationSurfacePeak',
                        'snr', 'slantRangeOffsetVariance', 'alongTrackOffsetVariance',
                        'crossOffsetVariance']
        for layer in layers:
            for fname in layer_fields:
                if fname in off_grp[layer]:
                    _register(f'{layer}/{fname}', off_grp[layer][fname])

    elif product == 'GCOV':
        cov_terms = []
        if 'listOfCovarianceTerms' in freq_grp:
            cov_terms = [t.decode() if isinstance(t, bytes) else str(t)
                         for t in freq_grp['listOfCovarianceTerms']]
        for term in cov_terms:
            if term in freq_grp:
                _register(term, freq_grp[term])
        for extra in ('mask', 'numberOfLooks', 'rtcGammaToSigmaFactor'):
            if extra in freq_grp:
                _register(extra, freq_grp[extra])

    if not loaders:
        h5.close()
        sys.exit(f'showimage: no displayable fields found in {filepath} ({product})')

    col_coords = None  # zeroDopplerTime per row
    row_coords = None  # slantRange per col
    _pt_map = {
        'RIFG': 'interferogram', 'RUNW': 'interferogram',
        'ROFF': 'pixelOffsets', 'GOFF': 'pixelOffsets',
        'GUNW': ('unwrappedInterferogram' if 'unwrappedInterferogram' in freq_grp
                 else 'wrappedInterferogram'),
    }
    _pt = _pt_map.get(product)
    if _pt and _pt in freq_grp:
        _cg = freq_grp[_pt]
        try:
            if 'zeroDopplerTime' in _cg:
                col_coords = np.array(_cg['zeroDopplerTime'])
        except Exception:
            pass
        try:
            if 'slantRange' in _cg:
                row_coords = np.array(_cg['slantRange'])
        except Exception:
            pass

    return nx, ny, product, loaders, h5, col_coords, row_coords


def hsvSpeedRender(speed, vmin=1.0, vmax=3000.0):
    """Log-scaled HSV speed rendering; replicates nisarBase2D.hsvSpeedRender.

    Returns float32 RGB array (ny, nx, 3) with values in [0, 1].
    Hue = log position in [vmin, vmax]; saturation fades to white below ~125 m/yr;
    NaN pixels are forced to white (saturation = 0).
    """
    from matplotlib import colors as mcolors
    background = np.isnan(speed)
    value = np.ones(speed.shape, dtype=np.float32)
    saturation = np.clip((speed / 125.0 + 0.5) / 1.5, 0, 1).astype(np.float32)
    saturation[background] = 0
    # denominator mixes log10(vmax) and natural log(vmin) — matches original
    hue = (np.log10(np.clip(speed, vmin, vmax)) /
           (np.log10(vmax) - np.log(vmin))).astype(np.float32)
    hue = np.nan_to_num(hue, nan=0.0)
    hsv = np.moveaxis(np.array([hue, saturation, value]), 0, 2)
    return mcolors.hsv_to_rgb(hsv).astype(np.float32)


def extractProfile(dec, r0, c0, r1, c1):
    """Sample dec along the line from (r0,c0) to (r1,c1) using bilinear interpolation."""
    length = max(int(np.hypot(r1 - r0, c1 - c0)), 1) + 1
    rows = np.linspace(r0, r1, length)
    cols = np.linspace(c0, c1, length)
    try:
        from scipy.ndimage import map_coordinates
        if dec.ndim == 2:
            vals = map_coordinates(np.nan_to_num(dec), [rows, cols], order=1, cval=np.nan)
        else:
            vals = np.stack(
                [map_coordinates(np.nan_to_num(dec[:, :, i]), [rows, cols],
                                 order=1, cval=np.nan)
                 for i in range(dec.shape[2])], axis=-1)
    except ImportError:
        # nearest-neighbour fallback
        ri = np.clip(np.round(rows).astype(int), 0, dec.shape[0] - 1)
        ci = np.clip(np.round(cols).astype(int), 0, dec.shape[1] - 1)
        vals = dec[ri, ci] if dec.ndim == 2 else dec[ri, ci, :]
    dist = np.linspace(0, np.hypot(c1 - c0, r1 - r0), length)
    return dist, vals


def openProfileWindow(dist, vals_or_list, p0, p1, titles=None, parent=None, pos=None):
    """Show profile(s) in a Toplevel window.

    vals_or_list: single ndarray or list of ndarrays (one per image).
    titles: optional list of per-subplot titles.
    """
    import tkinter as tk
    from tkinter import ttk
    import matplotlib.figure as mfig
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    vals_list = vals_or_list if isinstance(vals_or_list, list) else [vals_or_list]
    n = len(vals_list)
    if titles is None:
        titles = [None] * n

    WIN_W = 800
    WIN_H = 234 + 200 * n  # +34 for axis-control row

    win = tk.Toplevel()
    win.title(f'Profile  ({p0[1]}, {p0[0]}) → ({p1[1]}, {p1[0]})')

    if pos is not None:
        win.geometry(f'{WIN_W}x{WIN_H}+{pos[0]}+{pos[1]}')
    elif parent is not None:
        parent.update_idletasks()
        px = parent.winfo_x()
        pw = parent.winfo_width()
        sw2 = parent.winfo_screenwidth()
        sh2 = parent.winfo_screenheight()
        x = max(0, min(px + pw + 10, sw2 - WIN_W))
        y = max(0, (sh2 - WIN_H) // 2)
        win.geometry(f'{WIN_W}x{WIN_H}+{x}+{y}')
    else:
        win.geometry(f'{WIN_W}x{WIN_H}')

    fig = mfig.Figure(figsize=(8, WIN_H / DPI), dpi=DPI)
    prof_axes = []
    for i, (vals, ttl) in enumerate(zip(vals_list, titles)):
        ax = fig.add_subplot(n, 1, i + 1)
        prof_axes.append(ax)
        if vals.ndim == 1:
            ax.plot(dist, vals, color='steelblue')
        else:
            for j in range(vals.shape[1]):
                ax.plot(dist, vals[:, j], label=f'Band {j + 1}')
            ax.legend(fontsize=7)
        ax.set_xlabel('Distance (pixels)')
        ax.set_ylabel('Value')
        hdr = f'{ttl}: ' if ttl else ''
        ax.set_title(f'{hdr}col={p0[1]}, row={p0[0]}  →  col={p1[1]}, row={p1[0]}'
                     f'   ({dist[-1]:.1f} px)', fontsize=8)
        ax.grid(True, alpha=0.4)
    fig.tight_layout()

    btn_frame = ttk.Frame(win)
    btn_frame.pack(side='bottom', fill='x', padx=4, pady=(2, 6))
    ttk.Button(btn_frame, text='Close', command=win.destroy).pack(side='right', padx=4)

    ctrl_frame = ttk.Frame(win)
    ctrl_frame.pack(side='bottom', fill='x', padx=4, pady=2)
    log_y_p = [False]
    ymin_vp = tk.StringVar()
    ymax_vp = tk.StringVar()
    ttk.Label(ctrl_frame, text='Y:').pack(side='left')
    ttk.Entry(ctrl_frame, textvariable=ymin_vp, width=8).pack(side='left', padx=(0, 2))
    ttk.Label(ctrl_frame, text='to').pack(side='left')
    ttk.Entry(ctrl_frame, textvariable=ymax_vp, width=8).pack(side='left', padx=(0, 8))

    mpl_ref = [None]

    def _apply_prof_y():
        try:
            lo, hi = float(ymin_vp.get()), float(ymax_vp.get())
        except ValueError:
            return
        for ax in prof_axes:
            ax.set_ylim(lo, hi)
        mpl_ref[0].draw_idle()

    def _auto_prof_y():
        for ax in prof_axes:
            ax.set_ylim(auto=True)
            ax.relim()
            ax.autoscale_view()
        mpl_ref[0].draw_idle()
        if prof_axes:
            lo, hi = prof_axes[0].get_ylim()
            ymin_vp.set(f'{lo:.4g}')
            ymax_vp.set(f'{hi:.4g}')

    lbtn_ref = [None]

    def _toggle_prof_log():
        log_y_p[0] = not log_y_p[0]
        scale = 'log' if log_y_p[0] else 'linear'
        lbtn_ref[0].config(text='Log Y ✓' if log_y_p[0] else 'Log Y')
        for ax in prof_axes:
            ax.set_yscale(scale)
        mpl_ref[0].draw_idle()

    ttk.Button(ctrl_frame, text='Apply', command=_apply_prof_y).pack(side='left', padx=2)
    ttk.Button(ctrl_frame, text='Auto', command=_auto_prof_y).pack(side='left', padx=2)
    lbtn = ttk.Button(ctrl_frame, text='Log Y', command=_toggle_prof_log)
    lbtn.pack(side='left', padx=2)
    lbtn_ref[0] = lbtn

    mpl = FigureCanvasTkAgg(fig, master=win)
    mpl.draw()
    mpl.get_tk_widget().pack(fill='both', expand=True)
    mpl_ref[0] = mpl


# -----------------------------------------------------------------------
# Non-scroll path: single matplotlib figure with embedded colorbar
# -----------------------------------------------------------------------

def makeFigure(dec, title, cmap, vmin, vmax, is_rgb):
    """Build a matplotlib Figure sized exactly to the image in pixels."""
    import matplotlib.figure as mfig

    ny, nx = dec.shape[:2]
    fig_w_px = nx if is_rgb else nx + CBAR_PX
    fig = mfig.Figure(figsize=(fig_w_px / DPI, ny / DPI), dpi=DPI)

    if is_rgb:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(dec, interpolation='nearest', aspect='equal')
    else:
        cbar_frac = CBAR_PX / fig_w_px
        ax_right = 1 - cbar_frac - 0.02
        ax = fig.add_axes([0, 0, ax_right, 1])
        cax = fig.add_axes([ax_right + 0.03, 0.05, 0.04, 0.9])
        im = ax.imshow(dec, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        fig.colorbar(im, cax=cax)

    ax.set_title(title, fontsize=8)
    ax.axis('off')
    return fig, fig_w_px, ny


# -----------------------------------------------------------------------
# Scroll path: PhotoImage for fast panning + separate colorbar figure
# -----------------------------------------------------------------------

def decToPhoto(dec, cmap, vmin, vmax, is_rgb):
    """Render dec to a PIL PhotoImage using the matplotlib colormap."""
    from PIL import Image, ImageTk
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    if is_rgb:
        arr = (np.clip(dec, 0, 1) * 255).astype(np.uint8)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = mcm.get_cmap(cmap)(norm(np.nan_to_num(dec, nan=vmin)))
        arr = (rgba[:, :, :3] * 255).astype(np.uint8)

    return ImageTk.PhotoImage(Image.fromarray(arr))


def makeColorbarFig(cmap, vmin, vmax, height_px):
    """Standalone colorbar figure for the scroll layout."""
    import matplotlib.figure as mfig
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    fig = mfig.Figure(figsize=(CBAR_PX / DPI, height_px / DPI), dpi=DPI)
    cax = fig.add_axes([0.25, 0.05, 0.35, 0.9])
    sm = mcm.ScalarMappable(cmap=mcm.get_cmap(cmap),
                             norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    return fig


def _rebuildColorbar(ref, cmap, vmin, vmax):
    """Redraw a pane's standalone colorbar figure in place after a rescale/recolor."""
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    if ref['cbar_fig'] is None:
        return
    ref['cbar_fig'].clear()
    cax = ref['cbar_fig'].add_axes([0.25, 0.05, 0.35, 0.9])
    sm = mcm.ScalarMappable(cmap=mcm.get_cmap(cmap),
                             norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    ref['cbar_fig'].colorbar(sm, cax=cax)
    ref['cbar_cv'].draw()


def bindScroll(tk_canvas):
    """Bind mouse-wheel scroll for Windows/Mac and Linux."""
    def _y(event):
        tk_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
    def _x(event):
        tk_canvas.xview_scroll(int(-1 * (event.delta / 120)), 'units')
    tk_canvas.bind('<MouseWheel>', _y)
    tk_canvas.bind('<Shift-MouseWheel>', _x)
    tk_canvas.bind('<Button-4>', lambda e: tk_canvas.yview_scroll(-1, 'units'))
    tk_canvas.bind('<Button-5>', lambda e: tk_canvas.yview_scroll(1, 'units'))
    tk_canvas.bind('<Shift-Button-4>', lambda e: tk_canvas.xview_scroll(-1, 'units'))
    tk_canvas.bind('<Shift-Button-5>', lambda e: tk_canvas.xview_scroll(1, 'units'))


# -----------------------------------------------------------------------
# Main display
# -----------------------------------------------------------------------

def showImage(image_defs, sw, sh, switch_infos=None, rtl=False):
    """Display 1–3 images side by side with a floating control palette.

    image_defs: list of dicts, each with keys:
        dec (ndarray), title (str), cmap (str), vmin (float), vmax (float), is_rgb (bool)
    Images may differ in size; each pane gets its own scrollregion.
    rtl: if True, place the control palette at the right edge of the screen,
        with the image and plot/profile windows opening to its left
        (mirror of the default left-to-right layout) — lets a second
        instance be run without overlapping the first.
    """
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    n_imgs = len(image_defs)
    max_ny = max(d['dec'].shape[0] for d in image_defs)
    max_nx = max(d['dec'].shape[1] for d in image_defs)

    def _disp(arr, ol):
        """Flip array vertically for display when origin is lower (dy > 0)."""
        return np.flipud(arr) if ol else arr

    root = tk.Tk()
    root.title(' | '.join(f'{i+1}) {os.path.basename(d["title"])}'
                          for i, d in enumerate(image_defs)))

    # ---- command palette (separate floating window) ----
    palette = tk.Toplevel(root)
    palette.title('Controls')
    palette.resizable(False, False)
    palette.protocol('WM_DELETE_WINDOW', root.destroy)

    pick_active         = [False]
    profile_active      = [False]
    col_active          = [False]
    row_active          = [False]
    lines_visible       = [True]
    coords_active       = [any(d.get('col_coords') is not None for d in image_defs)]
    overlay_set_visible = [None]
    all_same_size = all(d['dec'].shape == image_defs[0]['dec'].shape for d in image_defs)
    scroll_synced = [all_same_size]  # default: synced iff all same size
    profile_pts         = []
    # per-mode state; each window is independent
    plot_states = {
        'col': {'win': None, 'axes': None, 'fig': None, 'canvas': None, 'single': False,
                'bias_vars': [], 'remove_mean': [False], 'history': []},
        'row': {'win': None, 'axes': None, 'fig': None, 'canvas': None, 'single': False,
                'bias_vars': [], 'remove_mean': [False], 'history': []},
    }

    btn_col = ttk.Frame(palette)
    btn_col.pack(side='top', fill='x', padx=4, pady=(4, 2))

    pick_btn    = ttk.Button(btn_col, text='Pick')
    profile_btn = ttk.Button(btn_col, text='Profile')
    col_btn     = ttk.Button(btn_col, text='Col Plot')
    row_btn     = ttk.Button(btn_col, text='Row Plot')
    lines_btn   = ttk.Button(btn_col, text='Lines ✓')
    coords_btn  = ttk.Button(btn_col, text='Coords ✓' if coords_active[0] else 'Coords')
    quit_btn    = ttk.Button(btn_col, text='Quit', command=root.destroy)
    core_btns = [pick_btn, profile_btn, col_btn, row_btn, lines_btn, coords_btn]
    if n_imgs > 1:
        sync_btn = ttk.Button(btn_col,
                              text='Sync ✓' if scroll_synced[0] else 'Sync')
        core_btns.append(sync_btn)
    else:
        sync_btn = None
    core_btns.append(quit_btn)
    for btn in core_btns:
        btn.pack(side='top', fill='x', pady=2, padx=2)

    ttk.Separator(btn_col, orient='horizontal').pack(fill='x', pady=(4, 2))
    ttk.Label(btn_col, text='Colormap:', anchor='w').pack(fill='x', padx=2)
    cmap_var = tk.StringVar(value=image_defs[0].get('cmap', 'gray'))
    cmap_combo = ttk.Combobox(btn_col, textvariable=cmap_var, values=CMAPS,
                               state='readonly', width=12)
    cmap_combo.pack(side='top', fill='x', pady=2, padx=2)

    ttk.Separator(btn_col, orient='horizontal').pack(fill='x', pady=(4, 2))
    mod_entries = []
    for _mi, _idef in enumerate(image_defs):
        if _idef['is_rgb']:
            mod_entries.append(None)
            continue
        _row_f = ttk.Frame(btn_col)
        _row_f.pack(fill='x', pady=1, padx=2)
        _lbl = f'P{_mi+1} mod:' if n_imgs > 1 else 'Mod:'
        ttk.Label(_row_f, text=_lbl, anchor='w').pack(side='left')
        _mv = _idef.get('mod_val')
        _var = tk.StringVar(value='' if _mv is None else str(_mv))
        _ent = ttk.Entry(_row_f, textvariable=_var, width=8)
        _ent.pack(side='left', fill='x', expand=True)
        mod_entries.append((_var, _ent))

    n_non_rgb = sum(1 for d in image_defs if not d['is_rgb'])
    common_vmin_var = tk.StringVar(value='')
    common_vmax_var = tk.StringVar(value='')
    common_scale_btn = None
    if n_non_rgb > 1:
        ttk.Separator(btn_col, orient='horizontal').pack(fill='x', pady=(4, 2))
        ttk.Label(btn_col, text='Common scale (min/max):', anchor='w').pack(fill='x', padx=2)
        _cs_row = ttk.Frame(btn_col)
        _cs_row.pack(fill='x', pady=1, padx=2)
        _cs_min_ent = ttk.Entry(_cs_row, textvariable=common_vmin_var, width=6)
        _cs_min_ent.pack(side='left', fill='x', expand=True, padx=(0, 2))
        _cs_max_ent = ttk.Entry(_cs_row, textvariable=common_vmax_var, width=6)
        _cs_max_ent.pack(side='left', fill='x', expand=True)
        common_scale_btn = ttk.Button(btn_col, text='Common Scale')
        common_scale_btn.pack(side='top', fill='x', pady=2, padx=2)

    status_var = tk.StringVar(value='Ready')
    status_lbl = ttk.Label(palette, textvariable=status_var, anchor='nw', wraplength=130)
    status_lbl.pack(side='bottom', fill='both', expand=True, padx=6, pady=(0, 4))

    def deactivate_all():
        pick_active[0] = False
        profile_active[0] = False
        col_active[0] = False
        row_active[0] = False
        pick_btn.config(text='Pick')
        profile_btn.config(text='Profile')
        col_btn.config(text='Col Plot')
        row_btn.config(text='Row Plot')

    def toggle_pick():
        if pick_active[0]:
            deactivate_all()
        else:
            deactivate_all()
            pick_active[0] = True
            pick_btn.config(text='Pick ●')
            profile_pts.clear()
    pick_btn.config(command=toggle_pick)

    def toggle_profile():
        if profile_active[0]:
            deactivate_all()
        else:
            deactivate_all()
            profile_active[0] = True
            profile_btn.config(text='Profile ●')
            profile_pts.clear()
            status_var.set('  Profile: click first point')
    profile_btn.config(command=toggle_profile)

    def toggle_col():
        if col_active[0]:
            deactivate_all()
        else:
            deactivate_all()
            col_active[0] = True
            col_btn.config(text='Col Plot ●')
            status_var.set('  Col Plot: click a pixel to plot that column')
    col_btn.config(command=toggle_col)

    def toggle_row():
        if row_active[0]:
            deactivate_all()
        else:
            deactivate_all()
            row_active[0] = True
            row_btn.config(text='Row Plot ●')
            status_var.set('  Row Plot: click a pixel to plot that row')
    row_btn.config(command=toggle_row)

    def toggle_lines():
        lines_visible[0] = not lines_visible[0]
        lines_btn.config(text='Lines ✓' if lines_visible[0] else 'Lines ✗')
        if overlay_set_visible[0]:
            overlay_set_visible[0](lines_visible[0])
    lines_btn.config(command=toggle_lines)

    def toggle_coords_palette():
        coords_active[0] = not coords_active[0]
        coords_btn.config(text='Coords ✓' if coords_active[0] else 'Coords')
    coords_btn.config(command=toggle_coords_palette)

    if sync_btn is not None:
        def toggle_sync():
            scroll_synced[0] = not scroll_synced[0]
            sync_btn.config(text='Sync ✓' if scroll_synced[0] else 'Sync')
        sync_btn.config(command=toggle_sync)

    def openOrReuseLineplot(mode):
        import matplotlib.figure as mfig
        state = plot_states[mode]
        WIN_W = 800
        BTN_H = 80
        x_label_idx = 'Row index' if mode == 'col' else 'Column index'

        def _cur_xlabel():
            if coords_active[0]:
                key = 'col_coord_label' if mode == 'col' else 'row_coord_label'
                lbl = image_defs[0].get(key) if image_defs else None
                return lbl if lbl else x_label_idx
            return x_label_idx

        def _is_alive():
            w = state['win']
            if w is None:
                return False
            try:
                return bool(w.winfo_exists())
            except Exception:
                return False

        def _build_axes(fig, single):
            fig.clf()
            xlabel = _cur_xlabel()
            if single:
                ax = fig.add_subplot(1, 1, 1)
                ax.grid(True, alpha=0.4)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Value')
                state['axes'] = [ax]
            else:
                state['axes'] = []
                for i, idef in enumerate(image_defs):
                    ax = fig.add_subplot(n_imgs, 1, i + 1)
                    ax.grid(True, alpha=0.4)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel('Value')
                    ax.set_title(idef['title'], fontsize=8)
                    state['axes'].append(ax)
            fig.tight_layout()

        def _clear_mode_overlays():
            items = col_overlay_items if mode == 'col' else row_overlay_items
            for cv, item in items:
                cv.delete(item)
            items.clear()

        if not _is_alive():
            single = state['single']
            WIN_H = 434 if single else 234 + 200 * n_imgs
            x, y = nextPlotGeometry(WIN_W, WIN_H)
            win = tk.Toplevel()
            win.title('Column Plots' if mode == 'col' else 'Row Plots')
            win.geometry(f'{WIN_W}x{WIN_H}+{x}+{y}')
            fig = mfig.Figure(figsize=(8, (WIN_H - BTN_H) / DPI), dpi=DPI)
            _build_axes(fig, single)

            combined_frame = ttk.Frame(win)
            combined_frame.pack(side='bottom', fill='x', padx=4, pady=4)
            combined_frame.columnconfigure(1, weight=1)

            # ---- col 0: action buttons ----
            btn_frame = ttk.Frame(combined_frame)
            btn_frame.grid(row=0, column=0, sticky='ns', padx=(0, 8))

            single_win_btn = ttk.Button(
                btn_frame, text='Single ✓' if single else 'Single')

            def toggle_single_win():
                state['single'] = not state['single']
                single_win_btn.config(
                    text='Single ✓' if state['single'] else 'Single')
                _clear_mode_overlays()
                new_h = 434 if state['single'] else 234 + 200 * n_imgs
                state['fig'].set_size_inches(8, max(1.0, (new_h - BTN_H) / DPI))
                _build_axes(state['fig'], state['single'])
                history = list(state.get('history', []))
                state['history'] = []
                replay_fn = doColPlot if mode == 'col' else doRowPlot
                for idx in history:
                    replay_fn(idx)
                apply_limits()
                state['canvas'].draw()
                if n_imgs > 1:
                    _update_y_rows()
                state['win'].geometry(f'{WIN_W}x{new_h}')

            single_win_btn.config(command=toggle_single_win)
            single_win_btn.pack(side='top', fill='x', pady=1)

            def save_plot():
                from tkinter import filedialog
                path = filedialog.asksaveasfilename(
                    parent=win,
                    defaultextension='.png',
                    filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'),
                               ('SVG', '*.svg'), ('All files', '*.*')])
                if path:
                    state['fig'].savefig(path, bbox_inches='tight')

            def clear_plots():
                seen = set()
                for ax in state['axes']:
                    if id(ax) in seen:
                        continue
                    seen.add(id(ax))
                    ax.cla()
                    ax.grid(True, alpha=0.4)
                    ax.set_xlabel(_cur_xlabel())
                    ax.set_ylabel('Value')
                state['history'] = []
                state['fig'].tight_layout()
                state['canvas'].draw_idle()
                _clear_mode_overlays()

            ttk.Button(btn_frame, text='Save', command=save_plot).pack(
                side='top', fill='x', pady=1)
            ttk.Button(btn_frame, text='Clear', command=clear_plots).pack(
                side='top', fill='x', pady=1)
            ttk.Button(btn_frame, text='Close', command=win.destroy).pack(
                side='top', fill='x', pady=1)

            # ---- col 1: range controls ----
            ctrl_frame = ttk.Frame(combined_frame)
            ctrl_frame.grid(row=0, column=1, sticky='nsew')
            log_y = [False]

            y_vars = [(tk.StringVar(), tk.StringVar()) for _ in range(n_imgs)]
            xmin_var = tk.StringVar()
            xmax_var = tk.StringVar()
            bias_vars = [tk.StringVar(value='0') for _ in range(n_imgs)]
            state['bias_vars'] = bias_vars

            if n_imgs > 1:
                # One Y row per subplot; show all in multi mode, only first in single mode
                y_rows_frames = []
                for idx in range(n_imgs):
                    ymin_v, ymax_v = y_vars[idx]
                    y_row = ttk.Frame(ctrl_frame)
                    lbl_w = ttk.Label(y_row, text=f'Y{idx + 1}:')
                    lbl_w.pack(side='left')
                    ttk.Entry(y_row, textvariable=ymin_v, width=8).pack(
                        side='left', padx=(0, 2))
                    ttk.Label(y_row, text='to').pack(side='left')
                    ttk.Entry(y_row, textvariable=ymax_v, width=8).pack(
                        side='left', padx=(0, 10))
                    y_rows_frames.append((y_row, lbl_w))
                x_row = ttk.Frame(ctrl_frame)
                ttk.Label(x_row, text='X:').pack(side='left')
                ttk.Entry(x_row, textvariable=xmin_var, width=8).pack(
                    side='left', padx=(0, 2))
                ttk.Label(x_row, text='to').pack(side='left')
                ttk.Entry(x_row, textvariable=xmax_var, width=8).pack(
                    side='left', padx=(0, 10))

                def _update_y_rows():
                    is_s = state['single']
                    for fr, lbl_w in y_rows_frames:
                        fr.pack_forget()
                    x_row.pack_forget()
                    for idx, (fr, lbl_w) in enumerate(y_rows_frames):
                        if is_s and idx > 0:
                            continue
                        lbl_w.config(text='Y:' if is_s else f'Y{idx + 1}:')
                        fr.pack(fill='x', pady=(0, 1))
                    x_row.pack(fill='x')

                _update_y_rows()
            else:
                # n_imgs == 1: Y and X on one row
                bot_row = ttk.Frame(ctrl_frame)
                bot_row.pack(fill='x')
                ymin_v0, ymax_v0 = y_vars[0]
                ttk.Label(bot_row, text='Y:').pack(side='left')
                ttk.Entry(bot_row, textvariable=ymin_v0, width=8).pack(
                    side='left', padx=(0, 2))
                ttk.Label(bot_row, text='to').pack(side='left')
                ttk.Entry(bot_row, textvariable=ymax_v0, width=8).pack(
                    side='left', padx=(0, 10))
                ttk.Label(bot_row, text='X:').pack(side='left')
                ttk.Entry(bot_row, textvariable=xmin_var, width=8).pack(
                    side='left', padx=(0, 2))
                ttk.Label(bot_row, text='to').pack(side='left')
                ttk.Entry(bot_row, textvariable=xmax_var, width=8).pack(
                    side='left', padx=(0, 10))

                def _update_y_rows():
                    pass  # no-op for single image

            # Bias row — always visible, one entry per image regardless of single/multi mode
            bias_row = ttk.Frame(ctrl_frame)
            bias_row.pack(fill='x', pady=(0, 1))
            for idx in range(n_imgs):
                lbl = f'B{idx + 1}:' if n_imgs > 1 else 'B:'
                ttk.Label(bias_row, text=lbl).pack(side='left')
                ttk.Entry(bias_row, textvariable=bias_vars[idx], width=6).pack(
                    side='left', padx=(0, 8))

            remove_mean = [False]
            state['remove_mean'] = remove_mean

            def apply_limits():
                axes = state['axes']
                is_single_ax = len(axes) == 1
                for i, ax in enumerate(axes):
                    ymin_v, ymax_v = y_vars[0] if is_single_ax else y_vars[i]
                    try:
                        ax.set_ylim(float(ymin_v.get()), float(ymax_v.get()))
                    except ValueError:
                        pass
                    try:
                        ax.set_xlim(float(xmin_var.get()), float(xmax_var.get()))
                    except ValueError:
                        pass
                state['canvas'].draw_idle()

            def auto_limits():
                axes = state['axes']
                is_single_ax = len(axes) == 1
                for ax in axes:
                    ax.set_ylim(auto=True)
                    ax.set_xlim(auto=True)
                    ax.relim()
                    ax.autoscale_view()
                state['canvas'].draw_idle()
                for i, ax in enumerate(axes):
                    lo, hi = ax.get_ylim()
                    ymin_v, ymax_v = y_vars[0] if is_single_ax else y_vars[i]
                    ymin_v.set(f'{lo:.4g}')
                    ymax_v.set(f'{hi:.4g}')
                if axes:
                    lo, hi = axes[0].get_xlim()
                    xmin_var.set(f'{lo:.4g}')
                    xmax_var.set(f'{hi:.4g}')

            log_btn_ref = [None]

            def toggle_log_y():
                log_y[0] = not log_y[0]
                scale = 'log' if log_y[0] else 'linear'
                log_btn_ref[0].config(text='Log Y ✓' if log_y[0] else 'Log Y')
                for ax in state['axes']:
                    ax.set_yscale(scale)
                state['canvas'].draw_idle()

            action_row = ttk.Frame(ctrl_frame)
            action_row.pack(fill='x', pady=(2, 0))
            ttk.Button(action_row, text='Apply', command=apply_limits).pack(
                side='left', padx=2)
            ttk.Button(action_row, text='Auto', command=auto_limits).pack(
                side='left', padx=2)
            log_btn = ttk.Button(action_row, text='Log Y', command=toggle_log_y)
            log_btn.pack(side='left', padx=2)
            log_btn_ref[0] = log_btn

            rm_btn_ref = [None]

            def toggle_remove_mean():
                remove_mean[0] = not remove_mean[0]
                rm_btn_ref[0].config(text='Rm Mean ✓' if remove_mean[0] else 'Rm Mean')
                history = list(state.get('history', []))
                if not history:
                    return
                seen = set()
                for i, ax in enumerate(state['axes'] or []):
                    if id(ax) in seen:
                        continue
                    seen.add(id(ax))
                    ax.cla()
                    ax.grid(True, alpha=0.4)
                    ax.set_xlabel(_cur_xlabel())
                    ax.set_ylabel('Value')
                    if not state['single'] and n_imgs > 1 and i < len(image_defs):
                        ax.set_title(image_defs[i]['title'], fontsize=8)
                state['history'] = []
                replay_fn = doColPlot if mode == 'col' else doRowPlot
                for idx in history:
                    replay_fn(idx)

            rm_btn = ttk.Button(action_row, text='Rm Mean', command=toggle_remove_mean)
            rm_btn.pack(side='left', padx=2)
            rm_btn_ref[0] = rm_btn

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            state.update({'win': win, 'fig': fig, 'canvas': canvas,
                          'bias_vars': bias_vars, 'remove_mean': remove_mean})

        return state['axes'], state['fig'], state['canvas']

    def doColPlot(col):
        axes, fig, canvas = openOrReuseLineplot('col')
        state = plot_states['col']
        state.setdefault('history', []).append(col)
        bias_vars_c = state.get('bias_vars') or []
        rm_mean = state.get('remove_mean', [False])[0]
        use_c = coords_active[0]
        colors = []
        single = state['single']
        bad_bias = []
        for i, idef in enumerate(image_defs):
            dec = idef.get('raw', idef['dec'])
            if col >= dec.shape[1]:
                colors.append(None)
                continue
            try:
                bias = float(bias_vars_c[i].get()) if i < len(bias_vars_c) else 0.0
            except ValueError:
                bias = 0.0
                bad_bias.append(i + 1)
            ax = axes[0] if single else axes[i]
            pfx = f'{i+1}: ' if single else ''
            rows_i = np.arange(dec.shape[0])
            if use_c:
                ccoords = idef.get('col_coords')
                x_arr = (ccoords if ccoords is not None and len(ccoords) == len(rows_i)
                         else rows_i)
            else:
                x_arr = rows_i
            if dec.ndim == 2:
                data = dec[:, col].astype(float)
                if rm_mean:
                    data -= np.nanmean(data)
                data += bias
                line, = ax.plot(x_arr, data, label=f'{pfx}col {col}')
                colors.append(line.get_color())
            else:
                plotted = []
                for j, ch in enumerate(('R', 'G', 'B')[:dec.shape[2]]):
                    data = dec[:, col, j].astype(float)
                    if rm_mean:
                        data -= np.nanmean(data)
                    data += bias
                    plotted.append(ax.plot(x_arr, data,
                                           label=f'{pfx}col {col} {ch}')[0])
                colors.append(plotted[0].get_color())
            ax.legend(fontsize=7)
        # Sync x-axis across subplots when all images have the same row count
        if not single and len(axes) > 1:
            sizes = [idef.get('raw', idef['dec']).shape[0] for idef in image_defs]
            if len(set(sizes)) == 1:
                active = [ax for ax in axes if ax.lines]
                if active:
                    x0 = min(ax.get_xlim()[0] for ax in active)
                    x1 = max(ax.get_xlim()[1] for ax in active)
                    for ax in axes:
                        ax.set_xlim(x0, x1)
        fig.tight_layout()
        canvas.draw_idle()
        msg = f'  plotted col {col}'
        if bad_bias:
            msg += f'  (invalid bias for P{",".join(map(str, bad_bias))} — using 0)'
        status_var.set(msg)
        return colors

    def doRowPlot(row):
        axes, fig, canvas = openOrReuseLineplot('row')
        state = plot_states['row']
        state.setdefault('history', []).append(row)
        bias_vars_r = state.get('bias_vars') or []
        rm_mean = state.get('remove_mean', [False])[0]
        use_c = coords_active[0]
        colors = []
        single = state['single']
        data_row = row
        bad_bias = []
        for i, idef in enumerate(image_defs):
            dec = idef.get('raw', idef['dec'])
            ol = idef.get('origin_lower', False)
            drow = (dec.shape[0] - 1 - row) if ol else row
            if i == 0:
                data_row = drow
            if drow >= dec.shape[0]:
                colors.append(None)
                continue
            try:
                bias = float(bias_vars_r[i].get()) if i < len(bias_vars_r) else 0.0
            except ValueError:
                bias = 0.0
                bad_bias.append(i + 1)
            ax = axes[0] if single else axes[i]
            pfx = f'{i+1}: ' if single else ''
            cols_i = np.arange(dec.shape[1])
            if use_c:
                rcoords = idef.get('row_coords')
                x_arr = (rcoords if rcoords is not None and len(rcoords) == len(cols_i)
                         else cols_i)
            else:
                x_arr = cols_i
            if dec.ndim == 2:
                data = dec[drow, :].astype(float)
                if rm_mean:
                    data -= np.nanmean(data)
                data += bias
                line, = ax.plot(x_arr, data, label=f'{pfx}row {drow}')
                colors.append(line.get_color())
            else:
                plotted = []
                for j, ch in enumerate(('R', 'G', 'B')[:dec.shape[2]]):
                    data = dec[drow, :, j].astype(float)
                    if rm_mean:
                        data -= np.nanmean(data)
                    data += bias
                    plotted.append(ax.plot(x_arr, data,
                                           label=f'{pfx}row {drow} {ch}')[0])
                colors.append(plotted[0].get_color())
            ax.legend(fontsize=7)
        # Sync x-axis across subplots when all images have the same col count
        if not single and len(axes) > 1:
            sizes = [idef.get('raw', idef['dec']).shape[1] for idef in image_defs]
            if len(set(sizes)) == 1:
                active = [ax for ax in axes if ax.lines]
                if active:
                    x0 = min(ax.get_xlim()[0] for ax in active)
                    x1 = max(ax.get_xlim()[1] for ax in active)
                    for ax in axes:
                        ax.set_xlim(x0, x1)
        fig.tight_layout()
        canvas.draw_idle()
        msg = f'  plotted row {row}'
        if bad_bias:
            msg += f'  (invalid bias for P{",".join(map(str, bad_bias))} — using 0)'
        status_var.set(msg)
        return colors

    def report_pick(col, row):
        val_parts = []
        data_row = row
        for i, idef in enumerate(image_defs):
            dec = idef.get('raw', idef['dec'])
            ol = idef.get('origin_lower', False)
            drow = (dec.shape[0] - 1 - row) if ol else row
            if i == 0:
                data_row = drow
            if 0 <= drow < dec.shape[0] and 0 <= col < dec.shape[1]:
                if dec.ndim == 2:
                    val_parts.append(f'val{i+1}={dec[drow, col]:.6g}')
                else:
                    val_parts.append(f'val{i+1}='
                                     + '/'.join(f'{v:.4g}' for v in dec[drow, col]))
            else:
                val_parts.append(f'val{i+1}=OOB')
        status_var.set(f'col={col}\nrow={data_row}\n' + '   '.join(val_parts))

    next_plot_y = [None]

    def nextPlotGeometry(win_w, win_h):
        root.update_idletasks()
        px = root.winfo_x()
        py = root.winfo_y()
        pw = root.winfo_width()
        if rtl:
            x = max(0, min(px - win_w - 10, sw - win_w))
        else:
            x = max(0, min(px + pw + 10, sw - win_w))
        if next_plot_y[0] is None:
            next_plot_y[0] = py
        y = max(0, min(next_plot_y[0], sh - win_h))
        next_plot_y[0] += win_h + 10
        if next_plot_y[0] + win_h > sh:
            next_plot_y[0] = py
        return x, y

    # ---- image area: N canvases side by side, synchronized scrolling ----
    palette.update_idletasks()
    PAL_W = palette.winfo_reqwidth()
    PAL_Y = 10
    if rtl:
        PAL_X = sw - PAL_W - 10
        IMG_X = None  # real value depends on win_w, set once it's known below
    else:
        PAL_X = 10
        IMG_X = PAL_X + PAL_W + 5
    IMG_Y = PAL_Y

    PLOT_WIN_W = 800
    cbar_w_total = sum(CBAR_PX if not d['is_rgb'] else 0 for d in image_defs)
    cbar_per_img = max(CBAR_PX if not d['is_rgb'] else 0 for d in image_defs)
    img_area_w = sw - PAL_W - PLOT_WIN_W - 35
    usable_h = sh - IMG_Y - DECO_H

    # Compute viewport dimensions for both stacking orientations, pick larger area
    vw_h = min(max_nx, max(50, (img_area_w - n_imgs * SCROLLBAR_W - cbar_w_total) // n_imgs))
    vh_h = min(max_ny, max(50, usable_h - LABEL_H - SCROLLBAR_W))
    vw_v = min(max_nx, max(50, img_area_w - SCROLLBAR_W - cbar_per_img))
    vh_v = min(max_ny, max(50, (usable_h - n_imgs * (SCROLLBAR_W + LABEL_H)) // n_imgs))
    stack_horiz = (n_imgs == 1) or (vw_h * vh_h >= vw_v * vh_v)
    viewport_w = vw_h if stack_horiz else vw_v
    viewport_h = vh_h if stack_horiz else vh_v

    all_canvases = []
    pane_refs = []  # per-pane refs for band switching

    def make_hscroll(this_c):
        def fn(*args):
            this_c.xview(*args)
            if scroll_synced[0]:
                for c in all_canvases:
                    if c is not this_c:
                        c.xview(*args)
        return fn

    def make_vscroll(this_c):
        def fn(*args):
            this_c.yview(*args)
            if scroll_synced[0]:
                for c in all_canvases:
                    if c is not this_c:
                        c.yview(*args)
        return fn

    outer = ttk.Frame(root)
    outer.pack(fill='both', expand=True)

    for i, idef in enumerate(image_defs):
        photo = decToPhoto(_disp(idef['dec'], idef.get('origin_lower', False)),
                           idef['cmap'], idef['vmin'], idef['vmax'], idef['is_rgb'])
        img_frame = ttk.Frame(outer)
        img_frame.pack(side='left' if stack_horiz else 'top', fill='both', expand=True)

        cbar_fig_ref = cbar_cv_ref = None
        if not idef['is_rgb']:
            cbar_fig_ref = makeColorbarFig(idef['cmap'], idef['vmin'], idef['vmax'], viewport_h)
            cbar_cv_ref  = FigureCanvasTkAgg(cbar_fig_ref, master=img_frame)
            cbar_cv_ref.draw()
            cbar_cv_ref.get_tk_widget().pack(side='right', fill='y')

        sf = ttk.Frame(img_frame)
        sf.pack(side='left', fill='both', expand=True)

        title_lbl = ttk.Label(sf, text=f'{i+1}) {os.path.basename(idef["title"])}',
                               anchor='center', wraplength=viewport_w)
        title_lbl.pack(side='top', fill='x', pady=(0, 1))

        xs = ttk.Scrollbar(sf, orient='horizontal')
        ys = ttk.Scrollbar(sf, orient='vertical')
        xs.pack(side='bottom', fill='x')
        ys.pack(side='right',  fill='y')

        c = tk.Canvas(sf, width=viewport_w, height=viewport_h,
                      xscrollcommand=xs.set, yscrollcommand=ys.set)
        c.pack(side='left', fill='both', expand=True)
        xs.config(command=make_hscroll(c))
        ys.config(command=make_vscroll(c))

        ny_i, nx_i = idef['dec'].shape[:2]
        img_item = c.create_image(0, 0, anchor='nw', image=photo)
        c.image = photo
        c.config(scrollregion=(0, 0, nx_i, ny_i))
        all_canvases.append(c)
        pane_refs.append({'canvas': c, 'img_item': img_item,
                          'cbar_fig': cbar_fig_ref, 'cbar_cv': cbar_cv_ref,
                          'title_lbl': title_lbl, 'ny': ny_i, 'nx': nx_i})

    def _wy(event):
        targets = all_canvases if scroll_synced[0] else [event.widget]
        for c in targets: c.yview_scroll(int(-1 * (event.delta / 120)), 'units')
    def _wx(event):
        targets = all_canvases if scroll_synced[0] else [event.widget]
        for c in targets: c.xview_scroll(int(-1 * (event.delta / 120)), 'units')
    def _b4(event):
        targets = all_canvases if scroll_synced[0] else [event.widget]
        for c in targets: c.yview_scroll(-1, 'units')
    def _b5(event):
        targets = all_canvases if scroll_synced[0] else [event.widget]
        for c in targets: c.yview_scroll(1, 'units')
    def _sb4(event):
        targets = all_canvases if scroll_synced[0] else [event.widget]
        for c in targets: c.xview_scroll(-1, 'units')
    def _sb5(event):
        targets = all_canvases if scroll_synced[0] else [event.widget]
        for c in targets: c.xview_scroll(1, 'units')
    for c in all_canvases:
        c.bind('<MouseWheel>',       _wy)
        c.bind('<Shift-MouseWheel>', _wx)
        c.bind('<Button-4>',         _b4)
        c.bind('<Button-5>',         _b5)
        c.bind('<Shift-Button-4>',   _sb4)
        c.bind('<Shift-Button-5>',   _sb5)

    # ---- overlay helpers (items stored as (canvas, item_id) pairs) ----
    profile_overlay_items = []
    col_overlay_items     = []   # vertical lines from Col Plot clicks
    row_overlay_items     = []   # horizontal lines from Row Plot clicks

    def clear_overlay():
        for canvas, item in profile_overlay_items:
            canvas.delete(item)
        profile_overlay_items.clear()

    def _add_canvas_item(canvas, item, lst):
        if not lines_visible[0]:
            canvas.itemconfigure(item, state='hidden')
        lst.append((canvas, item))

    def draw_marker(col, row, color='yellow'):
        r = 5
        for canvas in all_canvases:
            item = canvas.create_oval(col - r, row - r, col + r, row + r,
                                      outline=color, width=2)
            _add_canvas_item(canvas, item, profile_overlay_items)

    def draw_profile_line(c0, r0, c1, r1, color='yellow'):
        for canvas in all_canvases:
            item = canvas.create_line(c0, r0, c1, r1, fill=color, width=1, dash=(4, 2))
            _add_canvas_item(canvas, item, profile_overlay_items)

    def draw_col_line(col, colors):
        for ref, color in zip(pane_refs, colors):
            if color is None:
                continue
            item = ref['canvas'].create_line(col, 0, col, ref['ny'] - 1, fill=color, width=1)
            _add_canvas_item(ref['canvas'], item, col_overlay_items)

    def draw_row_line(row, colors):
        for ref, color in zip(pane_refs, colors):
            if color is None:
                continue
            item = ref['canvas'].create_line(0, row, ref['nx'] - 1, row, fill=color, width=1)
            _add_canvas_item(ref['canvas'], item, row_overlay_items)

    def canvas_set_visible(v):
        vis = 'normal' if v else 'hidden'
        for canvas, item in (profile_overlay_items
                              + col_overlay_items + row_overlay_items):
            canvas.itemconfigure(item, state=vis)
    overlay_set_visible[0] = canvas_set_visible

    # ---- click handler (bound to all canvases) ----
    def on_canvas_click(event):
        canvas = event.widget
        col = int(canvas.canvasx(event.x))
        row = int(canvas.canvasy(event.y))

        if pick_active[0]:
            report_pick(col, row)

        elif profile_active[0]:
            if len(profile_pts) == 0:
                clear_overlay()
                profile_pts.append((row, col))
                draw_marker(col, row)
                status_var.set(f'  Profile: first point col={col} row={row}'
                               f'  — click second point')
            else:
                profile_pts.append((row, col))
                draw_marker(col, row)
                p0, p1 = profile_pts
                draw_profile_line(p0[1], p0[0], p1[1], p1[0])
                dv = []
                for d in image_defs:
                    dec_d = d.get('raw', d['dec'])
                    ol_d = d.get('origin_lower', False)
                    dr0 = (dec_d.shape[0] - 1 - p0[0]) if ol_d else p0[0]
                    dr1 = (dec_d.shape[0] - 1 - p1[0]) if ol_d else p1[0]
                    dv.append(extractProfile(dec_d, dr0, p0[1], dr1, p1[1]))
                dist      = dv[0][0]
                vals_list = [x[1] for x in dv]
                titles    = [d['title'] for d in image_defs]
                prof_h    = 200 + 200 * n_imgs
                openProfileWindow(dist, vals_list, p0, p1, titles=titles,
                                  pos=nextPlotGeometry(800, prof_h))
                status_var.set(f'  Profile: ({p0[1]},{p0[0]}) → ({p1[1]},{p1[0]})'
                               f'  {dist[-1]:.1f} px — click to start new profile')
                profile_pts.clear()

        elif col_active[0]:
            colors = doColPlot(col)
            draw_col_line(col, colors)

        elif row_active[0]:
            colors = doRowPlot(row)
            draw_row_line(row, colors)

    for c in all_canvases:
        c.bind('<Button-1>', on_canvas_click)

    # ---- colormap selector callback ----
    def apply_cmap(event=None):
        new_cmap = cmap_var.get()
        for idef, ref in zip(image_defs, pane_refs):
            if idef['is_rgb']:
                continue
            idef['cmap'] = new_cmap
            new_photo = decToPhoto(_disp(idef.get('raw', idef['dec']),
                                         idef.get('origin_lower', False)),
                                   new_cmap, idef['vmin'], idef['vmax'], False)
            ref['canvas'].itemconfigure(ref['img_item'], image=new_photo)
            ref['canvas'].image = new_photo
            _rebuildColorbar(ref, new_cmap, idef['vmin'], idef['vmax'])
    cmap_combo.bind('<<ComboboxSelected>>', apply_cmap)

    # ---- common vmin/vmax across all non-RGB panes (toggle) ----
    common_scale_active = [False]
    _orig_scales = []

    def apply_common_scale(event=None):
        targets = [(idef, ref) for idef, ref in zip(image_defs, pane_refs)
                   if not idef['is_rgb']]
        if not targets:
            return

        def _redraw(idef, ref, vmin, vmax):
            new_photo = decToPhoto(_disp(idef.get('raw', idef['dec']),
                                         idef.get('origin_lower', False)),
                                   idef['cmap'], vmin, vmax, False)
            ref['canvas'].itemconfigure(ref['img_item'], image=new_photo)
            ref['canvas'].image = new_photo
            _rebuildColorbar(ref, idef['cmap'], vmin, vmax)

        if common_scale_active[0]:
            for (idef, ref), (ovmin, ovmax) in zip(targets, _orig_scales):
                idef['vmin'] = ovmin
                idef['vmax'] = ovmax
                _redraw(idef, ref, ovmin, ovmax)
            _orig_scales.clear()
            common_scale_active[0] = False
            common_scale_btn.config(text='Common Scale')
            status_var.set('Restored original scales')
            return

        vmin_str = common_vmin_var.get().strip()
        vmax_str = common_vmax_var.get().strip()
        try:
            vmin = (float(vmin_str) if vmin_str
                    else min(idef['vmin'] for idef, _ in targets))
            vmax = (float(vmax_str) if vmax_str
                    else max(idef['vmax'] for idef, _ in targets))
        except ValueError:
            status_var.set('Common scale: invalid min/max')
            return

        _orig_scales.clear()
        for idef, ref in targets:
            _orig_scales.append((idef['vmin'], idef['vmax']))
            idef['vmin'] = vmin
            idef['vmax'] = vmax
            _redraw(idef, ref, vmin, vmax)
        common_vmin_var.set(f'{vmin:.4g}')
        common_vmax_var.set(f'{vmax:.4g}')
        common_scale_active[0] = True
        common_scale_btn.config(text='Restore Scale')
        status_var.set(f'Common scale: [{vmin:.4g}, {vmax:.4g}]')

    if common_scale_btn is not None:
        common_scale_btn.config(command=apply_common_scale)
        _cs_min_ent.bind('<Return>', apply_common_scale)
        _cs_min_ent.bind('<KP_Enter>', apply_common_scale)
        _cs_max_ent.bind('<Return>', apply_common_scale)
        _cs_max_ent.bind('<KP_Enter>', apply_common_scale)

    # ---- mod applier (per pane) ----
    def make_mod_applier(p_idx):
        def apply_mod(event=None):
            entry_info = mod_entries[p_idx]
            if entry_info is None:
                return
            var, _ = entry_info
            val_str = var.get().strip()
            idef = image_defs[p_idx]
            ref = pane_refs[p_idx]
            if idef['is_rgb']:
                return
            base = idef.get('base', idef['dec'])
            try:
                mod_v = float(val_str) if val_str else None
            except ValueError:
                status_var.set(f'P{p_idx+1}: invalid mod')
                return
            dec = (np.where(np.isfinite(base), base % mod_v, base)
                   if mod_v is not None else base)
            vmin_a = idef.get('vmin_arg')
            vmax_a = idef.get('vmax_arg')
            vmin = vmin_a if vmin_a is not None else (
                0.0 if mod_v is not None else np.nanpercentile(dec, 2))
            vmax = vmax_a if vmax_a is not None else (
                mod_v if mod_v is not None else np.nanpercentile(dec, 98))
            idef['dec'] = dec
            idef['mod_val'] = mod_v
            idef['vmin'] = vmin
            idef['vmax'] = vmax
            new_photo = decToPhoto(_disp(dec, idef.get('origin_lower', False)),
                                   idef['cmap'], vmin, vmax, False)
            ref['canvas'].itemconfigure(ref['img_item'], image=new_photo)
            ref['canvas'].image = new_photo
            _rebuildColorbar(ref, idef['cmap'], vmin, vmax)
            status_var.set(f'P{p_idx+1}: mod={mod_v}')
        return apply_mod

    for _i, _ei in enumerate(mod_entries):
        if _ei is None:
            continue
        _, _ent = _ei
        _ap = make_mod_applier(_i)
        _ent.bind('<Return>', _ap)
        _ent.bind('<KP_Enter>', _ap)

    # ---- band switching (per pane) ----
    if switch_infos is not None and any(si is not None for si in switch_infos):
        ttk.Separator(btn_col, orient='horizontal').pack(fill='x', pady=(6, 2))

        def make_band_switcher(bname, bnum, p_idef, p_ref, p_si, p_idx):
            def switch():
                cache = p_si['cache']
                if cache is not None and bname in cache:
                    base_dec = cache[bname]
                else:
                    if 'ds' in p_si:
                        band_arr = readBand(p_si['ds'], bnum)
                        m_arr = p_si.get('mask_arr')
                        if m_arr is not None:
                            if m_arr.shape[:2] == band_arr.shape[:2]:
                                band_arr = band_arr.astype(np.float32, copy=True)
                                if p_si.get('mask_invert'):
                                    band_arr[m_arr != 0] = np.nan
                                else:
                                    band_arr[m_arr == 0] = np.nan
                            else:
                                print(f'showimage: mask shape {m_arr.shape[:2]} does not '
                                      f'match image shape {band_arr.shape[:2]} — skipping mask',
                                      file=sys.stderr)
                        base_dec = blockAverage(band_arr, p_si['factor'])
                    else:
                        base_dec = blockAverage(
                            p_si['loaders'][bname](), p_si['factor'])
                    if cache is not None:
                        cache[bname] = base_dec
                mod_v = p_si['mod_val']
                if mod_v is not None:
                    dec = np.where(np.isfinite(base_dec), base_dec % mod_v, base_dec)
                else:
                    dec = base_dec
                vmin_a, vmax_a = p_si['vmin_arg'], p_si['vmax_arg']
                vmin = vmin_a if vmin_a is not None else (
                    0.0 if mod_v is not None else np.nanpercentile(dec, 2))
                vmax = vmax_a if vmax_a is not None else (
                    mod_v if mod_v is not None else np.nanpercentile(dec, 98))
                p_idef.update({'dec': dec, 'base': base_dec,
                               'vmin': vmin, 'vmax': vmax, 'title': bname})
                new_photo = decToPhoto(_disp(dec, p_idef.get('origin_lower', False)),
                                       p_si['cmap'], vmin, vmax, False)
                p_ref['canvas'].itemconfigure(p_ref['img_item'], image=new_photo)
                p_ref['canvas'].image = new_photo
                _rebuildColorbar(p_ref, p_si['cmap'], vmin, vmax)
                p_ref['title_lbl'].config(text=f'{p_idx+1}) {bname}')
                if n_imgs == 1:
                    root.title(f'1) {bname}')
                # refresh the diff/sum pane if one exists
                d_idef = image_defs[-1]
                if d_idef.get('is_diff') and len(pane_refs) == len(image_defs):
                    d_ref = pane_refs[-1]
                    d_add = d_idef['add_mode']
                    d_op = '+' if d_add else '−'
                    d_arr, d_sc, d_stats = _computeDiffArray(
                        image_defs[0]['base'], image_defs[1]['base'], d_add)
                    d_title = f'P1 {d_op} P2    {d_stats}'
                    d_idef.update({'dec': d_arr, 'base': d_arr,
                                   'vmin': -d_sc, 'vmax': d_sc, 'title': d_title})
                    d_photo = decToPhoto(
                        _disp(d_arr, d_idef.get('origin_lower', False)),
                        'RdBu', -d_sc, d_sc, False)
                    d_ref['canvas'].itemconfigure(d_ref['img_item'], image=d_photo)
                    d_ref['canvas'].image = d_photo
                    _rebuildColorbar(d_ref, 'RdBu', -d_sc, d_sc)
                    d_ref['title_lbl'].config(text=f'{len(image_defs)}) {d_title}')
                status_var.set(f'Pane {p_idx+1} band: {bname}')
            return switch

        for p_idx, (idef, ref, si) in enumerate(zip(image_defs, pane_refs, switch_infos)):
            if si is None:
                continue
            if 'ds' in si:
                bnames = getBandNames(si['ds'])
            else:
                bnames = list(si['loaders'].keys())
            lbl = f'Bands ({p_idx+1})' if n_imgs > 1 else 'Bands'
            ttk.Label(btn_col, text=f'{lbl}:', anchor='w').pack(fill='x', padx=2, pady=(2, 0))
            for bnum, bname in enumerate(bnames, 1):
                ttk.Button(btn_col, text=bname,
                           command=make_band_switcher(bname, bnum, idef, ref, si, p_idx)).pack(
                    side='top', fill='x', pady=1, padx=2)

    # ---- position palette (right edge if rtl, else left), image window adjacent ----
    win_h_max = sh - IMG_Y - DECO_H
    if stack_horiz:
        win_w = n_imgs * (viewport_w + SCROLLBAR_W) + cbar_w_total
        win_h = min(LABEL_H + viewport_h + SCROLLBAR_W, win_h_max)
    else:
        win_w = viewport_w + SCROLLBAR_W + cbar_per_img
        win_h = min(n_imgs * (LABEL_H + viewport_h + SCROLLBAR_W), win_h_max)

    if rtl:
        IMG_X = max(0, PAL_X - 5 - win_w)
    status_lbl.config(wraplength=max(60, PAL_W - 12))
    palette.update_idletasks()
    pal_h = min(palette.winfo_reqheight(), win_h_max)
    palette.geometry(f'{PAL_W}x{pal_h}+{PAL_X}+{PAL_Y}')
    root.geometry(f'{win_w}x{win_h}+{IMG_X}+{IMG_Y}')

    root.mainloop()


def _computeDiffArray(a, b, add_mode):
    """Return (diff, scale, stats_str) for a P1-vs-P2 difference (or sum) pane."""
    op = '+' if add_mode else '−'
    diff = a + b if add_mode else a - b
    finite = diff[np.isfinite(diff)]
    scale = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    if scale == 0:
        scale = 1.0
    if finite.size:
        mean = float(np.nanmean(finite))
        std  = float(np.nanstd(finite))
        rms  = float(np.sqrt(np.nanmean(finite ** 2)))
        dmin = float(finite.min())
        dmax = float(finite.max())
        stats_str = (f'mean={mean:.4g}  std={std:.4g}  rms={rms:.4g}'
                     f'  min={dmin:.4g}  max={dmax:.4g}')
        print(f'Diff (P1 {op} P2):  {stats_str}  (n={finite.size:,})')
    else:
        stats_str = 'no valid pixels'
        print(f'Diff (P1 {op} P2):  no valid pixels')
    return diff, scale, stats_str


def _injectDiff(image_defs, add_mode):
    """Append a difference (or sum) pane from the first two entries of image_defs."""
    a = image_defs[0]['base']
    b = image_defs[1]['base']
    if a.shape != b.shape:
        sys.exit(f'--diff: image shapes differ ({a.shape} vs {b.shape}); '
                 'both inputs must be the same size')
    diff, scale, stats_str = _computeDiffArray(a, b, add_mode)
    op = '+' if add_mode else '−'
    image_defs.append({
        'dec': diff,
        'base': diff,
        'mod_val': None,
        'vmin_arg': None,
        'vmax_arg': None,
        'title': f'P1 {op} P2    {stats_str}',
        'cmap': 'RdBu',
        'vmin': -scale,
        'vmax': scale,
        'col_coords': image_defs[0].get('col_coords'),
        'row_coords': image_defs[0].get('row_coords'),
        'col_coord_label': image_defs[0].get('col_coord_label'),
        'row_coord_label': image_defs[0].get('row_coord_label'),
        'origin_lower': image_defs[0].get('origin_lower', False),
        'is_rgb': False,
        'is_diff': True,
        'add_mode': add_mode,
    })


def resolveFilename(f):
    """Return (resolved_path, is_geodat).

    If f exists and GDAL can open it, return (f, False).
    Otherwise try, in priority order, f+'.vrt', f+'.tif', f+'.h5'/'.he5'/'.hdf5'
    (NISAR), then check for f+'.geodat' sidecar (geoimage scalar: data in f,
    metadata in f.geodat).
    Exits with an error if nothing is found.
    """
    if os.path.exists(f):
        if gdal is None:
            return f, False
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        try:
            ds = gdal.Open(f)
        except Exception:
            ds = None
        finally:
            gdal.PopErrorHandler()
        if ds is not None:
            ds = None
            return f, False
    for ext in ('.vrt', '.tif', '.h5', '.he5', '.hdf5'):
        cand = f + ext
        if os.path.exists(cand):
            return cand, False
    if os.path.exists(f + '.geodat'):
        return f, True
    sys.exit(f'showimage: cannot find {f!r} '
             f'(tried {f}.vrt, {f}.tif, {f}.h5; geodat sidecar {f}.geodat not found)')


def geodatToGdalMem(f):
    """Read a GrIMP scalar geodat binary image and return a GDAL MEM dataset."""
    try:
        from utilities.geoimage import geoimage as Geoimage
    except ImportError:
        sys.exit('showimage: utilities.geoimage not available — cannot read geodat files')
    gi = Geoimage(verbose=False)
    try:
        gi.readData(f, geoType='scalar')
    except Exception as exc:
        sys.exit(f'showimage: cannot read geodat image {f!r}: {exc}')
    arr = np.asarray(gi.x, dtype=np.float32)
    ny, nx = arr.shape
    x0 = float(gi.xx[0]) * 1000.0
    dx = float(gi.xx[1] - gi.xx[0]) * 1000.0 if nx > 1 else 1.0
    y0 = float(gi.yy[0]) * 1000.0
    dy = float(gi.yy[1] - gi.yy[0]) * 1000.0 if ny > 1 else -1.0
    gt = (x0, dx, 0.0, y0, 0.0, dy)
    driver = gdal.GetDriverByName('MEM')
    mem_ds = driver.Create('', nx, ny, 1, gdal.GDT_Float32)
    mem_ds.SetGeoTransform(gt)
    band = mem_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(-2e9)
    return mem_ds


def main():
    parser = argparse.ArgumentParser(
        description='Display 1–3 same-size VRT or GeoTIFF images side by side.',
        epilog='Part of the utilities package.')
    parser.add_argument('files', metavar='FILE', nargs='+',
                        help='Input image(s) (.vrt, .tif, .tiff) — up to 3')
    parser.add_argument('--cmap', default='gray',
                        help='Colormap for single-band images (default: gray)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Lower clip value (default: 2nd percentile)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Upper clip value (default: 98th percentile)')
    parser.add_argument('-vmin', type=float, default=None, dest='vmin',
                        help=argparse.SUPPRESS)
    parser.add_argument('-vmax', type=float, default=None, dest='vmax',
                        help=argparse.SUPPRESS)
    parser.add_argument('--decFactor', type=int, default=None,
                        help='Decimation factor (default: auto-fit to screen)')
    parser.add_argument('--fullRes', action='store_true',
                        help='Display at full resolution (equivalent to --decFactor 1)')
    parser.add_argument('-decFactor', type=int, default=None, dest='decFactor',
                        help=argparse.SUPPRESS)
    parser.add_argument('--vel', action='store_true',
                        help='Read vx+vy from a 2-band VRT; display speed = sqrt(vx²+vy²) '
                             'mod 100 (use --mod to override the modulus)')
    parser.add_argument('--mod', type=float, default=None, metavar='X',
                        help='Display image modulo X '
                             '(default: 100 with --vel, off otherwise)')
    parser.add_argument('--log', action='store_true',
                        help='With --vel: log-scaled HSV rendering (vmin=1, vmax=3000 m/yr; '
                             'override with --vmin/--vmax)')
    parser.add_argument('--bands', nargs='+', metavar='BAND', default=None,
                        help='Show 1–3 named bands from a single file (by band description)')
    parser.add_argument('--freq', default='frequencyA',
                        help='NISAR HDF5 frequency band (default: frequencyA)')
    parser.add_argument('--pol', default=None,
                        help='NISAR HDF5 polarization (default: auto-detect first available)')
    parser.add_argument('--noCache', action='store_true',
                        help='Disable decimated-band cache (reduces memory use; '
                             're-reads from disk on each band switch)')
    parser.add_argument('--right', action='store_true',
                        help='Place the control palette at the right edge of the '
                             'screen, with the image and plot/profile windows '
                             'opening to its left (mirrors the default left-to-right '
                             'layout) — lets a second instance run without '
                             'overlapping the first')
    _diff_group = parser.add_mutually_exclusive_group()
    _diff_group.add_argument('--diff', action='store_true',
                        help='Show two images and their difference as a third pane '
                             '(pane 3 = pane 1 − pane 2, RdBu colormap, auto-scaled '
                             'symmetrically). Requires exactly 2 files or 2 --bands.')
    _diff_group.add_argument('--add', action='store_true',
                        help='Show two images and their sum as a third pane '
                             '(pane 3 = pane 1 + pane 2, RdBu colormap, auto-scaled '
                             'symmetrically). Requires exactly 2 files or 2 --bands.')
    _mask_group = parser.add_mutually_exclusive_group()
    _mask_group.add_argument('--mask', default=None, metavar='MASKFILE',
                        help='Single-band mask file; pixels where the mask is 0 are '
                             'set to noData in all displayed images')
    _mask_group.add_argument('--invMask', default=None, metavar='MASKFILE',
                        help='Single-band mask file; pixels where the mask is non-zero '
                             'are set to noData in all displayed images '
                             '(opposite of --mask)')
    args = parser.parse_args()

    if args.vel and len(args.files) != 1:
        sys.exit('--vel requires exactly one input file')
    if args.log and not args.vel:
        sys.exit('--log requires --vel')
    if args.bands and args.vel:
        sys.exit('--bands and --vel are mutually exclusive')
    if args.bands and len(args.files) != 1:
        sys.exit('--bands requires exactly one input file')
    if args.bands and len(args.bands) > 3:
        sys.exit('--bands: at most 3 band names allowed')
    args.combine = args.diff or args.add
    if args.combine:
        if args.vel:
            sys.exit('--diff/--add is not compatible with --vel')
        n_src = len(args.bands) if args.bands else len(args.files)
        if n_src != 2:
            sys.exit('--diff/--add requires exactly 2 sources (2 files or --bands with 2 band names)')
    if not args.vel and not args.bands and not args.combine and len(args.files) > 3:
        sys.exit('showimage: at most 3 files can be displayed simultaneously')

    # Resolve filenames: fall back to .vrt, .tif, .geodat if bare name not found
    resolved_files = []
    geodat_flags = []
    for f in args.files:
        rpath, is_gd = resolveFilename(f)
        if rpath != f:
            print(f'showimage: {f!r} not found, using {rpath!r}')
        resolved_files.append(rpath)
        geodat_flags.append(is_gd)
    args.files = resolved_files

    nisar_exts = {'.h5', '.he5', '.hdf5'}
    n_nisar = sum(1 for f in args.files if os.path.splitext(f)[1].lower() in nisar_exts)
    if 0 < n_nisar < len(args.files):
        sys.exit('showimage: cannot mix NISAR HDF5 and non-HDF5 files')
    is_nisar = n_nisar > 0

    if is_nisar:
        if args.vel:
            sys.exit('showimage: --vel is not supported for NISAR HDF5 files')
        if args.mask or args.invMask:
            sys.exit('showimage: --mask/--invMask are not supported for NISAR HDF5 files')

        sw, sh = getScreenSize()
        nisar_infos = []
        for f in args.files:
            nxi, nyi, product, loaders, h5, col_c, row_c = openNisarH5(
                f, frequency=args.freq, pol=args.pol)
            nisar_infos.append((f, nxi, nyi, product, loaders, h5, col_c, row_c))

        n_display = len(args.bands) if args.bands else len(args.files)
        if args.combine:
            n_display += 1
        # factor for --bands (single file); multi-file uses per-file factor inside loop
        nxi0, nyi0 = nisar_infos[0][1], nisar_infos[0][2]
        if args.fullRes:
            factor = 1
        elif args.decFactor is not None:
            factor = max(1, args.decFactor)
        else:
            factor = max(1, math.ceil(nxi0 * n_display / sw), math.ceil(nyi0 / sh))

        mod_val = args.mod
        image_defs = []
        switch_infos = []

        if args.bands:
            # Display up to 3 named fields; no band-switcher buttons (same as GDAL --bands)
            f, nxi, nyi, product, loaders, h5, col_c, row_c = nisar_infos[0]
            for bname in args.bands:
                if bname not in loaders:
                    print(f'showimage: band "{bname}" not found — skipping',
                          file=sys.stderr)
                    print(f'  Available: {", ".join(loaders.keys())}',
                          file=sys.stderr)
                    continue
                base_dec = blockAverage(loaders[bname](), factor)
                dec = (np.where(np.isfinite(base_dec), base_dec % mod_val, base_dec)
                       if mod_val is not None else base_dec)
                vmin = args.vmin if args.vmin is not None else (
                    0.0 if mod_val is not None else np.nanpercentile(dec, 2))
                vmax = args.vmax if args.vmax is not None else (
                    mod_val if mod_val is not None else np.nanpercentile(dec, 98))
                print(f'{f} [{bname}]: {nxi}×{nyi} px, decimation ×{factor}')
                ny_dec, nx_dec = base_dec.shape[:2]
                _col_c = (np.linspace(col_c[0], col_c[-1], ny_dec)
                          if col_c is not None and len(col_c) > 0 else None)
                _row_c = (np.linspace(row_c[0], row_c[-1], nx_dec)
                          if row_c is not None and len(row_c) > 0 else None)
                image_defs.append({
                    'dec': dec,
                    'base': base_dec,
                    'mod_val': mod_val,
                    'vmin_arg': args.vmin,
                    'vmax_arg': args.vmax,
                    'title': bname,
                    'cmap': args.cmap,
                    'vmin': vmin,
                    'vmax': vmax,
                    'col_coords': _col_c,
                    'row_coords': _row_c,
                    'col_coord_label': 'Azimuth Time (s)',
                    'row_coord_label': 'Slant Range (m)',
                    'origin_lower': product.startswith('R'),
                    'is_rgb': False,
                })
            if not image_defs:
                for _f, _nxi, _nyi, _prod, _loaders, h5, _cc, _rc in nisar_infos:
                    h5.close()
                sys.exit('showimage: no valid bands found')
            switch_infos = None
        else:
            for f, nxi, nyi, product, loaders, h5, col_c, row_c in nisar_infos:
                if args.fullRes:
                    fi = 1
                elif args.decFactor is not None:
                    fi = max(1, args.decFactor)
                else:
                    fi = max(1, math.ceil(nxi * n_display / sw), math.ceil(nyi / sh))
                first_band = next(iter(loaders))
                base_dec = blockAverage(loaders[first_band](), fi)
                dec = (np.where(np.isfinite(base_dec), base_dec % mod_val, base_dec)
                       if mod_val is not None else base_dec)
                vmin = args.vmin if args.vmin is not None else (
                    0.0 if mod_val is not None else np.nanpercentile(dec, 2))
                vmax = args.vmax if args.vmax is not None else (
                    mod_val if mod_val is not None else np.nanpercentile(dec, 98))
                print(f'{f} [{product}]: {nxi}×{nyi} px, {len(loaders)} field(s), '
                      f'decimation ×{fi}')
                print(f'  Displaying: {first_band}')
                print(f'  Available: {", ".join(loaders.keys())}')
                ny_dec_i, nx_dec_i = dec.shape[:2]
                _col_c = (np.linspace(col_c[0], col_c[-1], ny_dec_i)
                          if col_c is not None and len(col_c) > 0 else None)
                _row_c = (np.linspace(row_c[0], row_c[-1], nx_dec_i)
                          if row_c is not None and len(row_c) > 0 else None)
                image_defs.append({
                    'dec': dec,
                    'base': base_dec,
                    'mod_val': mod_val,
                    'vmin_arg': args.vmin,
                    'vmax_arg': args.vmax,
                    'title': f'{first_band}: {os.path.basename(f)}',
                    'cmap': args.cmap,
                    'vmin': vmin,
                    'vmax': vmax,
                    'col_coords': _col_c,
                    'row_coords': _row_c,
                    'col_coord_label': 'Azimuth Time (s)',
                    'row_coord_label': 'Slant Range (m)',
                    'origin_lower': product.startswith('R'),
                    'is_rgb': False,
                })
                switch_infos.append({
                    'loaders': loaders,
                    'factor': fi,
                    'mod_val': mod_val,
                    'cmap': args.cmap,
                    'vmin_arg': args.vmin,
                    'vmax_arg': args.vmax,
                    'cache': None if args.noCache else {first_band: base_dec},
                })

        if args.combine:
            _injectDiff(image_defs, args.add)
            if switch_infos is not None:
                switch_infos = list(switch_infos) + [None]
        showImage(image_defs, sw, sh, switch_infos=switch_infos, rtl=args.right)
        for _f, _nxi, _nyi, _prod, _loaders, h5, _cc, _rc in nisar_infos:
            h5.close()
        return

    if gdal is None:
        sys.exit('osgeo.gdal not available — install gdal')

    gdal.UseExceptions()

    datasets = []
    for f, is_geodat in zip(args.files, geodat_flags):
        try:
            datasets.append(geodatToGdalMem(f) if is_geodat else gdal.Open(f))
        except Exception as e:
            sys.exit(f'Cannot open {f}: {e}')

    mask_arr = None
    mask_invert = False
    if args.mask or args.invMask:
        mask_file = args.mask or args.invMask
        mask_invert = args.invMask is not None
        mask_file, mask_is_geodat = resolveFilename(mask_file)
        try:
            mask_ds = geodatToGdalMem(mask_file) if mask_is_geodat else gdal.Open(mask_file)
        except Exception as e:
            sys.exit(f'Cannot open mask file {mask_file}: {e}')
        mask_arr = mask_ds.GetRasterBand(1).ReadAsArray()
        mask_ds = None

    def applyMask(full_arr):
        """Set masked pixels to NaN in a full-resolution array (before decimation)."""
        if mask_arr is None:
            return full_arr
        if mask_arr.shape[:2] != full_arr.shape[:2]:
            print(f'showimage: mask shape {mask_arr.shape[:2]} does not match '
                  f'image shape {full_arr.shape[:2]} — skipping mask',
                  file=sys.stderr)
            return full_arr
        out = full_arr.astype(np.float32, copy=True)
        if mask_invert:
            out[mask_arr != 0] = np.nan
        else:
            out[mask_arr == 0] = np.nan
        return out

    sizes = [(ds.RasterXSize, ds.RasterYSize) for ds in datasets]
    nx, ny = sizes[0]  # used by --vel and --bands (single-file paths)
    sw, sh = getScreenSize()
    n_display = 1 if args.vel else (len(args.bands) if args.bands else len(args.files))
    if args.combine:
        n_display += 1
    _inject_raw = (args.mod is not None and not args.vel and not args.bands
                   and not args.combine and len(args.files) == 1)
    if _inject_raw:
        n_display += 1

    if args.fullRes:
        factor = 1
    elif args.decFactor is not None:
        factor = max(1, args.decFactor)
    else:
        factor = max(1, math.ceil(nx * n_display / sw), math.ceil(ny / sh))

    mod_val = args.mod
    image_defs = []

    if args.vel:
        ds = datasets[0]
        f  = args.files[0]
        _gt_v = ds.GetGeoTransform()
        _ol = _gt_v[5] > 0 or isNisarOriginLowerProduct(f)
        if ds.RasterCount < 2:
            sys.exit('--vel: file must have at least 2 bands (vx, vy)')
        if mod_val is None:
            mod_val = 100.0
        vx = applyMask(readBand(ds, 1))
        vy = applyMask(readBand(ds, 2))
        speed = np.where(np.isfinite(vx) & np.isfinite(vy),
                         np.sqrt(vx**2 + vy**2), np.nan)
        dec = blockAverage(speed, factor)
        _ny_d, _nx_d = dec.shape[:2]
        _col_c_v = _gt_v[3] + np.arange(_ny_d) * _gt_v[5] * factor
        _row_c_v = _gt_v[0] + np.arange(_nx_d) * _gt_v[1] * factor
        if args.log:
            sv_min = args.vmin if args.vmin is not None else 1.0
            sv_max = args.vmax if args.vmax is not None else 3000.0
            raw = dec.copy()
            dec = hsvSpeedRender(dec, sv_min, sv_max)
            print(f'{f}: {nx}×{ny} px, speed log HSV {sv_min}–{sv_max} m/yr, '
                  f'decimation ×{factor}')
            image_defs.append({
                'dec': dec,
                'raw': raw,
                'base': raw,
                'mod_val': None,
                'vmin_arg': args.vmin,
                'vmax_arg': args.vmax,
                'title': f'speed log HSV: {f}',
                'cmap': args.cmap,
                'vmin': None,
                'vmax': None,
                'col_coords': _col_c_v,
                'row_coords': _row_c_v,
                'col_coord_label': 'Y coordinate',
                'row_coord_label': 'X coordinate',
                'origin_lower': _ol,
                'is_rgb': True,
            })
        else:
            base_dec = dec.copy()
            dec = np.where(np.isfinite(base_dec), base_dec % mod_val, base_dec)
            vmin = args.vmin if args.vmin is not None else 0.0
            vmax = args.vmax if args.vmax is not None else mod_val
            print(f'{f}: {nx}×{ny} px, speed from bands 1+2, mod {mod_val:.4g}, '
                  f'decimation ×{factor}')
            image_defs.append({
                'dec': dec,
                'base': base_dec,
                'mod_val': mod_val,
                'vmin_arg': args.vmin,
                'vmax_arg': args.vmax,
                'title': f'speed mod {mod_val:.4g}: {f}',
                'cmap': args.cmap,
                'vmin': vmin,
                'vmax': vmax,
                'col_coords': _col_c_v,
                'row_coords': _row_c_v,
                'col_coord_label': 'Y coordinate',
                'row_coord_label': 'X coordinate',
                'origin_lower': _ol,
                'is_rgb': False,
            })
    elif args.bands:
        ds = datasets[0]
        f  = args.files[0]
        _gt_b = ds.GetGeoTransform()
        _ol = _gt_b[5] > 0 or isNisarOriginLowerProduct(f)
        for bname in args.bands:
            bnum = findBandByName(ds, bname)
            if bnum is None:
                print(f'showimage: band "{bname}" not found in {f} — skipping',
                      file=sys.stderr)
                continue
            base_dec = blockAverage(applyMask(readBand(ds, bnum)), factor)
            dec = (np.where(np.isfinite(base_dec), base_dec % mod_val, base_dec)
                   if mod_val is not None else base_dec)
            vmin = args.vmin if args.vmin is not None else (
                0.0 if mod_val is not None else np.nanpercentile(dec, 2))
            vmax = args.vmax if args.vmax is not None else (
                mod_val if mod_val is not None else np.nanpercentile(dec, 98))
            print(f'{f} [{bname}]: {nx}×{ny} px, decimation ×{factor}')
            _ny_d, _nx_d = base_dec.shape[:2]
            _col_c = _gt_b[3] + np.arange(_ny_d) * _gt_b[5] * factor
            _row_c = _gt_b[0] + np.arange(_nx_d) * _gt_b[1] * factor
            image_defs.append({
                'dec': dec,
                'base': base_dec,
                'mod_val': mod_val,
                'vmin_arg': args.vmin,
                'vmax_arg': args.vmax,
                'title': bname,
                'cmap': args.cmap,
                'vmin': vmin,
                'vmax': vmax,
                'col_coords': _col_c,
                'row_coords': _row_c,
                'col_coord_label': 'Y coordinate',
                'row_coord_label': 'X coordinate',
                'origin_lower': _ol,
                'is_rgb': False,
            })
        if not image_defs:
            sys.exit('showimage: no valid bands found')
    else:
        factors_per_file = []
        for ds, f in zip(datasets, args.files):
            nx_i, ny_i = ds.RasterXSize, ds.RasterYSize
            _gt_i = ds.GetGeoTransform()
            _ol = _gt_i[5] > 0 or isNisarOriginLowerProduct(f)
            if args.fullRes:
                fi = 1
            elif args.decFactor is not None:
                fi = max(1, args.decFactor)
            else:
                fi = max(1, math.ceil(nx_i * n_display / sw), math.ceil(ny_i / sh))
            factors_per_file.append(fi)
            nb = ds.RasterCount
            print(f'{f}: {nx_i}×{ny_i} px, {nb} band(s), decimation ×{fi}')

            if nb > 1:
                bnames = getBandNames(ds)
                suggestion = ' '.join(bnames[:3])
                print(f'  Displaying band 1.  To select bands:')
                print(f'    showimage [options] {os.path.basename(f)} --bands {suggestion}')
                print(f'  Available bands: {", ".join(bnames)}')

            base_dec = blockAverage(applyMask(readBand(ds, 1)), fi)
            dec = (np.where(np.isfinite(base_dec), base_dec % mod_val, base_dec)
                   if mod_val is not None else base_dec)
            vmin = args.vmin if args.vmin is not None else (
                0.0 if mod_val is not None else np.nanpercentile(dec, 2))
            vmax = args.vmax if args.vmax is not None else (
                mod_val if mod_val is not None else np.nanpercentile(dec, 98))
            _ny_d, _nx_d = base_dec.shape[:2]
            _col_c = _gt_i[3] + np.arange(_ny_d) * _gt_i[5] * fi
            _row_c = _gt_i[0] + np.arange(_nx_d) * _gt_i[1] * fi

            image_defs.append({
                'dec': dec,
                'base': base_dec,
                'mod_val': mod_val,
                'vmin_arg': args.vmin,
                'vmax_arg': args.vmax,
                'title': f,
                'cmap': args.cmap,
                'vmin': vmin,
                'vmax': vmax,
                'col_coords': _col_c,
                'row_coords': _row_c,
                'col_coord_label': 'Y coordinate',
                'row_coord_label': 'X coordinate',
                'origin_lower': _ol,
                'is_rgb': False,
            })

    if _inject_raw and image_defs:
        idef0 = image_defs[0]
        base0 = idef0['base']
        image_defs.append({
            'dec': base0,
            'base': base0,
            'mod_val': None,
            'vmin_arg': args.vmin,
            'vmax_arg': args.vmax,
            'title': idef0['title'],
            'cmap': args.cmap,
            'vmin': (args.vmin if args.vmin is not None
                     else float(np.nanpercentile(base0, 2))),
            'vmax': (args.vmax if args.vmax is not None
                     else float(np.nanpercentile(base0, 98))),
            'col_coords': idef0.get('col_coords'),
            'row_coords': idef0.get('row_coords'),
            'col_coord_label': idef0.get('col_coord_label'),
            'row_coord_label': idef0.get('row_coord_label'),
            'origin_lower': idef0.get('origin_lower', False),
            'is_rgb': False,
        })

    switch_infos = None
    if not args.vel and not args.bands:
        per_pane = [
            {'ds': ds, 'factor': fi, 'mod_val': mod_val,
             'cmap': args.cmap, 'vmin_arg': args.vmin, 'vmax_arg': args.vmax,
             'cache': None if args.noCache else {},
             'mask_arr': mask_arr, 'mask_invert': mask_invert}
            if ds.RasterCount > 1 else None
            for ds, fi in zip(datasets, factors_per_file)
        ]
        if any(si is not None for si in per_pane):
            switch_infos = per_pane

    if _inject_raw and switch_infos is not None:
        raw_si = {**switch_infos[0], 'mod_val': None,
                  'cache': None if args.noCache else {}}
        switch_infos = [switch_infos[0], raw_si]

    if args.combine:
        _injectDiff(image_defs, args.add)
        if switch_infos is not None:
            switch_infos = list(switch_infos) + [None]

    showImage(image_defs, sw, sh, switch_infos=switch_infos, rtl=args.right)


def showvel():
    """CLI entry point: showimage --vel (reads vx+vy VRT, displays speed)."""
    sys.argv = [sys.argv[0], '--vel'] + sys.argv[1:]
    main()


def showoffsets():
    """CLI entry point: showimage FILE [opts] --bands RangeOffsets AzimuthOffsets Correlation."""
    sys.argv = ([sys.argv[0]]
                + sys.argv[1:]
                + ['--bands', 'RangeOffsets', 'AzimuthOffsets', 'Correlation'])
    main()


if __name__ == '__main__':
    main()
