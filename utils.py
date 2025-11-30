# utils.py
# Helpers for reading openEMS-style HDF5 dumps and converting TD -> FD.
# Requires: h5py, numpy

from __future__ import annotations
from typing import Optional, Tuple, Sequence, Callable, Dict, Any, List
import numpy as np
import h5py
from dataclasses import dataclass
import matplotlib.pyplot as plt
import skrf as rf


@dataclass
class FieldDump:
    time: Optional[np.ndarray]  # (Nt,) seconds
    dt: Optional[float]  # seconds
    x: np.ndarray  # (Nx,) meters
    y: np.ndarray  # (Ny,) meters
    z: np.ndarray  # (Nz,) meters
    F_td: np.ndarray  # (Nt, Nx, Ny, Nz, 3)


def read_hdf5_dump(path: str, tick_period: float = 1.0) -> FieldDump:
    """
    Reads layout:
      /Mesh/x,y,z
      /FieldData/TD/<frame>  with shape (3, Nz, Ny, Nx)
    Interprets <frame> as an integer tick (timestamp/index).
    time = ticks * tick_period; dt = median(diff(ticks)) * tick_period
    """
    with h5py.File(path, "r") as f:
        x = np.asarray(f["/Mesh/x"][()], dtype=float)
        y = np.asarray(f["/Mesh/y"][()], dtype=float)
        z = np.asarray(f["/Mesh/z"][()], dtype=float)

        td_group = f["/FieldData/TD"]
        # sort numerically by key name (e.g., "00000049")
        keys = sorted(td_group.keys(), key=lambda k: int(k))
        ticks = np.array([int(k) for k in keys], dtype=float)

        first = np.asarray(td_group[keys[0]][()])  # (3, Nz, Ny, Nx)
        Nx, Ny, Nz = first.shape[3], first.shape[2], first.shape[1]
        Nt = len(keys)

        E_td = np.empty((Nt, Nx, Ny, Nz, 3), dtype=first.dtype)
        for i, k in enumerate(keys):
            F = np.asarray(td_group[k][()])  # (3, Nz, Ny, Nx)
            E_td[i] = np.transpose(F, (3, 2, 1, 0))  # -> (Nx, Ny, Nz, 3)

        # Build time & dt from ticks
        time = ticks * float(tick_period)
        dt = float(np.median(np.diff(time))) if time.size > 1 else None

    return FieldDump(time=time, dt=dt, x=x, y=y, z=z, F_td=E_td)


# ---------------------- TD → FD CONVERSION ----------------------

def td_to_fd_dft(E_td: np.ndarray,
                 time: Optional[np.ndarray],
                 dt: Optional[float],
                 f_hz: float) -> np.ndarray:
    """
    Single-frequency DFT of the time-domain field.

    Parameters
    ----------
    E_td : np.ndarray
        Shape (Nt, Nx, Ny, Nz, 3).
    time : np.ndarray | None
        If provided, used directly (supports irregular spacing).
    dt : float | None
        If `time` is None, uniform sample period (seconds).
    f_hz : float
        Target frequency.

    Returns
    -------
    np.ndarray
        Complex spectrum at f_hz, shape (Nx, Ny, Nz, 3).

    Notes
    -----
    Uses Riemann-sum DFT: sum_t E(t) * exp(-j2π f t) * Δt
    """
    if E_td.ndim != 5 or E_td.shape[-1] != 3:
        raise ValueError("E_td must have shape (Nt, Nx, Ny, Nz, 3).")

    Nt = E_td.shape[0]
    if time is None:
        if dt is None:
            raise ValueError("Provide either `time` or `dt`.")
        time = np.arange(Nt, dtype=float) * float(dt)
        delta_t = float(dt)
    else:
        time = np.asarray(time, dtype=float)
        if time.shape[0] != Nt:
            raise ValueError("`time` length does not match E_td.")
        # approximate Δt if nonuniform
        delta_t = float(np.median(np.diff(time)))

    t = time.reshape(Nt, 1, 1, 1, 1)
    kernel = np.exp(-1j * 2.0 * np.pi * f_hz * t)
    E_fd = np.sum(E_td * kernel, axis=0) * delta_t  # (Nx, Ny, Nz, 3)
    return E_fd


def td_to_fd_fft(E_td: np.ndarray, dt: float, f_hz: float) -> np.ndarray:
    """
    FFT-based pick of the nearest positive-frequency bin.

    Parameters
    ----------
    E_td : np.ndarray
        Shape (Nt, Nx, Ny, Nz, 3).
    dt : float
        Sample period (seconds), uniform.
    f_hz : float
        Target frequency.

    Returns
    -------
    np.ndarray
        Complex spectrum at the closest FFT bin, shape (Nx, Ny, Nz, 3).

    Notes
    -----
    Faster than DFT, but requires uniform sampling. If you need exact
    interpolation between bins, consider a Goertzel or zoom-FFT.
    """
    Nt = E_td.shape[0]
    freqs = np.fft.rfftfreq(Nt, d=dt)  # positive freqs
    spec = np.fft.rfft(E_td, axis=0) * dt  # (Nf, Nx, Ny, Nz, 3); scaling ~ integral
    k = int(np.argmin(np.abs(freqs - f_hz)))
    return spec[k]


# ---------------------- SMALL UTILITIES ----------------------

def extract_slice(E_fd: np.ndarray, z_coords: np.ndarray, z_value: float):
    """
    Extract (Ex, Ey, Ez) at the z closest to z_value.

    Parameters
    ----------
    E_fd : np.ndarray
        Shape (Nx, Ny, Nz, 3), complex.
    z_coords : np.ndarray
        Array of z positions (same units as z_value, e.g. meters).
    z_value : float
        Desired z position (same units as z_coords).

    Returns
    -------
    Ex, Ey, Ez : np.ndarray
        2D arrays of shape (Nx, Ny), complex.
    z_index : int
        Index actually used.
    z_actual : float
        z coordinate actually used.
    """
    if E_fd.ndim != 4 or E_fd.shape[-1] != 3:
        raise ValueError("E_fd must have shape (Nx, Ny, Nz, 3)")

    # find closest z index
    idx = int(np.argmin(np.abs(z_coords - z_value)))
    z_actual = float(z_coords[idx])

    Ex = E_fd[:, :, idx, 0]
    Ey = E_fd[:, :, idx, 1]
    Ez = E_fd[:, :, idx, 2]
    return Ex, Ey, Ez, idx, z_actual


def inspect_h5(path: str, max_depth: int = 6) -> None:
    """
    Print a simple tree of the HDF5 file to help discover dataset paths.
    """

    def _walk(name, obj, depth):
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}- {name}  [dataset] shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"{indent}+ {name}  [group]")
            if depth < max_depth:
                for k, v in obj.items():
                    _walk(f"{name}/{k}" if name != "/" else f"/{k}", v, depth + 1)

    with h5py.File(path, "r") as f:
        _walk("/", f["/"], 0)


# --- utils.py additions (flexible projection) ---

def _build_mask_from_rects(x_mm: np.ndarray,
                           y_mm: np.ndarray,
                           rects_mm: List[Dict[str, float]]) -> np.ndarray:
    """
    rects_mm: list of {'x1':..., 'x2':..., 'y1':..., 'y2':...} in mm, axis-aligned
    Returns mask_inside with shape (Nx, Ny): True where inside ANY rect.
    """
    xx, yy = np.meshgrid(x_mm, y_mm, indexing="ij")
    mask = np.zeros_like(xx, dtype=bool)
    for r in rects_mm:
        m = (xx >= r['x1']) & (xx <= r['x2']) & (yy >= r['y1']) & (yy <= r['y2'])
        mask |= m
    return mask


def plot_ez_2d(
        fd,
        E_fd: np.ndarray,
        z_value: float,
        func: Callable[[np.ndarray], np.ndarray] = np.abs,
        func_str: str = "Ez",
        cmap: str = "jet",
        clim: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        *,
        rects_mm: Optional[List[Dict[str, float]]] = None,  # <- add geometry here (mm)
        outside_color: Optional[str] = "lightgray",  # <- color for outside
        outside_mode: str = "mask",  # "mask" -> NaN with set_bad,
        # "value" -> force 'outside_value'
        outside_value: float = np.nan
):
    """
    2D colormap of Ez at z_value (meters). X horizontal, Y vertical (mm).
    If rects_mm provided (list of rectangles in mm), plot only inside geometry;
    outside is shown with 'outside_color' (mask mode) or set to 'outside_value'.
    """
    Ex, Ey, Ez, idx, z_used = extract_slice(E_fd, fd.z, z_value)
    Z = func(Ez)

    # coords to mm
    x_mm = fd.x * 1e3
    y_mm = fd.y * 1e3
    z_used_mm = z_used * 1e3

    # build geometry mask if rectangles are given
    mask_inside = None
    if rects_mm is not None and len(rects_mm) > 0:
        mask_inside = _build_mask_from_rects(x_mm, y_mm, rects_mm)

    # prepare colormap (copy so we can tweak)
    cm = plt.get_cmap(cmap).copy()
    if outside_color is not None:
        cm.set_bad(outside_color)  # for NaNs
        cm.set_under(outside_color)  # if we use masked arrays + clim

    # apply masking/value outside geometry
    if mask_inside is not None:
        if outside_mode == "mask":
            Zplot = np.ma.array(Z, mask=~mask_inside)
        elif outside_mode == "value":
            Z2 = Z.copy()
            Z2[~mask_inside] = outside_value
            Zplot = Z2
        else:
            raise ValueError("outside_mode must be 'mask' or 'value'")
    else:
        Zplot = Z

    # meshgrid for plotting
    xx, yy = np.meshgrid(x_mm, y_mm, indexing="ij")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    pc = ax.pcolormesh(xx, yy, Zplot, shading="auto", cmap=cm)
    if clim is not None:
        pc.set_clim(*clim)
    plt.colorbar(pc, ax=ax)

    if title is None:
        title = f"{func_str} slice at z = {z_used_mm:.2f} mm (idx {idx})"
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    return pc, {"z_index": idx, "z_used_mm": z_used_mm}


def plot_ez_line_y(
        fd,
        E_fd: np.ndarray,
        z_value: float,
        x_value: Optional[float] = None,
        func: Callable[[np.ndarray], np.ndarray] = np.abs,
        func_str: str = "Ez",
        clim: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        y_lines_mm: Optional[np.ndarray] = None
):
    """
    1D cut of Ez along X at z_value (meters) and fixed y_value (meters).
    X = horizontal axis. Coordinates shown in mm.
    """
    Ex, Ey, Ez, z_idx, z_used = extract_slice(E_fd, fd.z, z_value)

    Z = func(Ez)

    # pick y index
    if x_value is None:
        x_idx = Z.shape[0] // 2
    else:
        x_idx = int(np.argmin(np.abs(fd.x - x_value)))
    x_used = float(fd.x[x_idx])

    # convert coords to mm
    x_used_mm = x_used * 1e3
    y_mm = fd.y * 1e3
    z_used_mm = z_used * 1e3

    cut_vals = Z[x_idx, :]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    line = ax.plot(y_mm, cut_vals, marker="o", label=label or f"x={x_used_mm:.2f} mm")
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("projection of Ez")
    ax.set_title(
        f"{func_str} line cut along Y at x={x_used_mm:.2f} mm, z={z_used_mm:.2f} mm \n"
        f"(idx x={x_idx}, z={z_idx})"
    )
    ax.grid(True)
    if label:
        ax.legend()

    if clim:
        ax.set_ylim(clim[0], clim[1])

    # optional vertical marker lines at specified y (meters)
    if y_lines_mm is not None:
        ys = np.atleast_1d(np.asarray(y_lines_mm, dtype=float))
        for y_mark_mm in ys:
            vl = ax.axvline(y_mark_mm, linestyle="--", linewidth=1.0, alpha=0.6)

    return line[0], {
        "z_index": z_idx, "z_used_mm": z_used_mm,
        "x_index": x_idx, "x_used_mm": x_used_mm
    }


def plot_js_2d(
        fd,
        J_fd: np.ndarray,
        z_value: float,
        func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = np.abs,
        func_str: str = "Js",
        cmap: str = "jet",
        clim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None
):
    """
    2D colormap of Ez at z_value (meters). X horizontal, Y vertical (mm).
    If rects_mm provided (list of rectangles in mm), plot only inside geometry;
    outside is shown with 'outside_color' (mask mode) or set to 'outside_value'.
    """
    Jx, Jy, Jz, idx, z_used = extract_slice(J_fd, fd.z, z_value)
    Z = func(Jx, Jy, Jz)

    # coords to mm
    x_mm = fd.x * 1e3
    y_mm = fd.y * 1e3
    z_used_mm = z_used * 1e3

    # prepare colormap (copy so we can tweak)
    cm = plt.get_cmap(cmap).copy()

    # meshgrid for plotting
    xx, yy = np.meshgrid(x_mm, y_mm, indexing="ij")

    fig, ax = plt.subplots(figsize=(7, 6))

    pc = ax.pcolormesh(xx, yy, Z, shading="auto", cmap=cm)
    if clim is not None:
        pc.set_clim(*clim)
    plt.colorbar(pc, ax=ax)

    if title is None:
        title = f"{func_str} slice at z = {z_used_mm:.3f} mm (idx {idx})"
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    return pc, {"z_index": idx, "z_used_mm": z_used_mm}


def plot_smith_skrf(
        s11: np.ndarray,
        freq_hz: Optional[np.ndarray] = None,
        idx_main: Optional[int] = None,
        idx_res: Optional[int] = None,
        *,
        patch_width_mm: Optional[float] = None,
        patch_length_mm: Optional[float] = None,
        patchR: Optional[float] = None,
        patchG: Optional[float] = None,
        patchR2: Optional[float] = None,
        patchG2: Optional[float] = None,
        point_color: str = "k",
        z0: float = 50.0,
        ax: Optional[plt.Axes] = None,
        title: str = "Smith Chart",
) -> Tuple[plt.Axes, Dict[str, Any]]:
    """
    Plot S11 on a Smith chart using scikit-rf (no fallback).

    Parameters
    ----------
    s11 : ndarray of complex, shape (N,)
        Reflection coefficient Γ vs frequency.
    freq_hz : ndarray of float, optional
        Frequency array in Hz aligned with s11; used for labels.
    idx_main : int, optional
        Index to highlight (circle marker), e.g. current cursor frequency.
    idx_res : int, optional
        Index to highlight (square marker), e.g. resonance.
    patch_width_mm, patch_length_mm, len180_mm : float, optional
        Geometry values for legend (mm).
    patchR, patchG, patchR2, patchG2 : float, optional
        R/G values for legend.
    point_color : str
        Color for the markers (default 'k').
    z0 : float
        Reference impedance for Network creation (Ω).
    ax : matplotlib Axes, optional
        Axes to draw on. New figure if None.
    title : str
        Title string.

    Returns
    -------
    ax : matplotlib.axes.Axes
    meta : dict
        Useful info: handles, chosen freqs, etc.

    Notes
    -----
    Requires: `scikit-rf` (pip install scikit-rf).
    """
    import skrf as rf  # fail fast if not installed

    s11 = np.asarray(s11)
    if s11.ndim != 1:
        raise ValueError("s11 must be a 1D complex array of reflection coefficients.")

    # Build a dummy/real Frequency for a 1-port Network, then plot using scikit-rf.
    if freq_hz is None:
        # If no frequency provided, create a linear dummy axis (units don't matter for plotting).
        freq = rf.Frequency(1, len(s11), len(s11), unit="Hz")
        f_for_labels = None
    else:
        freq_hz = np.asarray(freq_hz, dtype=float)
        if freq_hz.shape[0] != s11.shape[0]:
            raise ValueError("freq_hz must have same length as s11.")
        freq = rf.Frequency.from_f(freq_hz, unit="Hz")
        f_for_labels = freq_hz

    # Create 1-port Network from Γ data
    s = s11.reshape(-1, 1, 1)  # (N, 1, 1)
    ntw = rf.Network(frequency=freq, s=s, z0=z0)

    # Make axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Plot S11 on Smith using scikit-rf (draws grid + data)
    ntw.plot_s_smith(m=0, n=0, ax=ax, label="S11")

    # Add markers
    legend_handles = []
    legend_labels = []

    # Replace the auto legend entry with a custom one containing geometry
    # (scikit-rf's call already added a legend; we'll rebuild it cleanly)
    ax.legend().remove() if ax.get_legend() else None

    # 1) Main S11 label with geometry values
    if (patch_width_mm is not None) or (patch_length_mm is not None):
        pw = "?" if patch_width_mm is None else f"{patch_width_mm:.2f}"
        pl = "?" if patch_length_mm is None else f"{patch_length_mm:.2f}"
        base_label = f"S11 (Patch W={pw} mm, L={pl} mm)"
    else:
        base_label = "S11"
    # Create an invisible handle to carry the label
    trace_handle, = ax.plot([], [], color="C0")
    legend_handles.append(trace_handle)
    legend_labels.append(base_label)

    f_main = f_res = None
    handle_main = handle_res = None

    if idx_main is not None:
        idx_main = int(idx_main)
        g = s11[idx_main]
        handle_main, = ax.plot([np.real(g)], [np.imag(g)],
                               marker="o", linestyle="None", color=point_color)
        if f_for_labels is not None:
            f_main = float(f_for_labels[idx_main])
        parts = []
        if f_main is not None: parts.append(f"{f_main / 1e9:.2f} GHz")
        parts.append(f"S11={g:.3f}")
        if patchR is not None: parts.append(f"R={patchR:.2f}")
        if patchG is not None: parts.append(f"Gnorm={patchG:.2f}")
        legend_handles.append(handle_main)
        legend_labels.append(", ".join(parts))

    if idx_res is not None:
        idx_res = int(idx_res)
        g2 = s11[idx_res]
        handle_res, = ax.plot([np.real(g2)], [np.imag(g2)],
                              marker="s", linestyle="None", color=point_color)
        if f_for_labels is not None:
            f_res = float(f_for_labels[idx_res])
        parts = []
        if f_res is not None: parts.append(f"{f_res / 1e9:.2f} GHz")
        parts.append(f"S11={g2:.3f}")
        if patchR2 is not None: parts.append(f"R2={patchR2:.2f}")
        if patchG2 is not None: parts.append(f"G2norm={patchG2:.2f}")
        legend_handles.append(handle_res)
        legend_labels.append(", ".join(parts))

    ax.legend(legend_handles, legend_labels, loc="upper right", frameon=True)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    return ax, {
        "network": ntw,
        "idx_main": idx_main,
        "idx_res": idx_res,
        "freq_main_hz": f_main,
        "freq_res_hz": f_res,
        "handle_main": handle_main,
        "handle_res": handle_res,
    }


def plot_directivity_linear_polar(theta, nf2ff_res, freq):
    """
    Plot absolute directivity (linear scale) in polar coordinates.
    The main lobe will peak at Dmax (linear).

    Parameters
    ----------
    theta : array-like
        Angle values in degrees.
    nf2ff_res : object
        Result object from openEMS CalcNF2FF, must have .E_norm and .Dmax.
    freq : float
        Frequency for inserting to plot captions.
    """
    theta_rad = np.deg2rad(theta)

    # Angular power distribution and total radiated power
    # P_rad shape: (nfreq, ntheta, nphi)
    P_rad = np.abs(nf2ff_res.P_rad[0])       # ensure non-negative
    Prad_tot = np.real(nf2ff_res.Prad[0])    # scalar

    # Directivity in linear scale: D = 4π * P_rad / Prad
    D_lin = 4.0 * np.pi * P_rad / Prad_tot

    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot(theta_rad, np.squeeze(D_lin[:, 0]), linewidth=2, label="xz-plane")
    ax.plot(theta_rad, np.squeeze(D_lin[:, 1]), linewidth=2, label="yz-plane")

    # 0° at top, clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_title(
        f"Frequency: {freq / 1e9:.3f} GHz — Directivity (linear). "
        f"Dmax: {np.max(D_lin):.3f}"
    )
    ax.grid(True)
    ax.legend(loc="lower right")
    plt.tight_layout()


def plot_directivity_db_polar(theta, nf2ff_res, freq):
    """
    Plot absolute directivity (dB) in polar coordinates using
    P_rad (angular power distribution) and Prad (total radiated power).

    Parameters
    ----------
    theta : array-like
        Theta angle values in degrees (nf2ff_res.theta).
    nf2ff_res : object
        Result object from openEMS CalcNF2FF.
        Must provide: P_rad, Prad, Dmax.
    freq : float
        Frequency for inserting to plot captions.
    """
    theta_rad = np.deg2rad(theta)

    # Angular power distribution and total radiated power
    # P_rad shape: (nfreq, ntheta, nphi)
    P_rad = np.abs(nf2ff_res.P_rad[0])       # ensure non-negative
    Prad_tot = np.real(nf2ff_res.Prad[0])    # scalar

    # Directivity in linear scale: D = 4π * P_rad / Prad
    D_lin = 4.0 * np.pi * P_rad / Prad_tot

    # Avoid log10(0) -> -inf
    D_lin = np.maximum(D_lin, 1e-20)

    # Convert to dB
    D_dB = 10.0 * np.log10(D_lin)

    # xz-plane and yz-plane cuts (assuming phi indices 0 and 1)
    D_xz = np.squeeze(D_dB[:, 0])
    D_yz = np.squeeze(D_dB[:, 1])

    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot(theta_rad, D_xz, linewidth=2, label="xz-plane (phi=0°)")
    ax.plot(theta_rad, D_yz, linewidth=2, label="yz-plane (phi=90°)")

    # 0° at top, clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    Dmax_dB = 10 * np.log10(np.max(D_lin))
    # Optionally compare with nf2ff_res.Dmax[fi]
    Dmax_from_res = 10 * np.log10(nf2ff_res.Dmax[0])

    ax.set_title(
        f"f = {freq/1e9:.3f} GHz — Directivity (dB)\n"
        f"Dmax (integrated) ≈ {Dmax_dB:.2f} dB,  nf2ff Dmax = {Dmax_from_res:.2f} dB"
    )

    ax.grid(True)
    ax.legend(loc="lower right")
    plt.tight_layout()


def plot_directivity_db(theta, nf2ff_res, freq):
    """
    Plot directivity in dBi (Cartesian x–y plot).

    Parameters
    ----------
    theta : array-like
        Angle values in degrees.
    nf2ff_res : object
        Result object from openEMS CalcNF2FF, must have .E_norm and .Dmax.
    freq : float
        Frequency for inserting to plot captions.
    """
    # Angular power distribution and total radiated power
    # P_rad shape: (nfreq, ntheta, nphi)
    P_rad = np.abs(nf2ff_res.P_rad[0])       # ensure non-negative
    Prad_tot = np.real(nf2ff_res.Prad[0])    # scalar

    # Directivity in linear scale: D = 4π * P_rad / Prad
    D_lin = 4.0 * np.pi * P_rad / Prad_tot

    # Avoid log10(0) -> -inf
    D_lin = np.maximum(D_lin, 1e-20)

    # Convert to dB
    D_dB = 10.0 * np.log10(D_lin)

    # xz-plane and yz-plane cuts (assuming phi indices 0 and 1)
    D_xz = np.squeeze(D_dB[:, 0])
    D_yz = np.squeeze(D_dB[:, 1])

    plt.figure()
    plt.plot(theta, np.squeeze(D_xz), "k-", linewidth=2, label="xz-plane")
    plt.plot(theta, np.squeeze(D_yz), "r--", linewidth=2, label="yz-plane")

    plt.grid(True)
    plt.ylabel("Directivity (dBi)")
    plt.xlabel("Theta/Phi (deg)")
    plt.title(f"Frequency: {freq / 1e9:.3f} GHz")
    plt.legend()
    plt.tight_layout()


def plot_s11(freq: np.ndarray,
             s11_dB: np.ndarray,
             *,
             level_db: float = -10.0,
             annotate: bool = True,
             ax: Optional[plt.Axes] = None,
             title = "Reflection Coefficient $S_{11}$") -> Dict[str, Optional[float]]:
    """
    Plot S11 vs frequency and annotate the deepest minimum and the
    -10 dB bandwidth around that minimum.

    Parameters
    ----------
    freq : array-like
        Frequency array (Hz).
    s11_dB : array-like
        S11 values in dB (negative for reflections < 0 dB).
    level_db : float
        Threshold level for bandwidth detection (default: -10 dB).
    annotate : bool
        If True, draw markers/lines/labels on the plot.
    ax : matplotlib.axes.Axes or None
        Existing axes to plot on. Creates a new figure if None.
    title : str
        The optional title for the plot.

    Returns
    -------
    info : dict
        {
          "f0_Hz": center frequency at deepest minimum,
          "S11_min_dB": value at the minimum,
          "f_low_Hz": lower -10 dB crossing (None if not found),
          "f_high_Hz": upper -10 dB crossing (None if not found),
          "BW_Hz": bandwidth (f_high - f_low) or None,
          "FBW": fractional BW = BW / f0 or None,
          "Q": f0 / BW (narrowband approx) or None
        }
    """
    # Make clean 1D arrays and sort by frequency
    f = np.asarray(freq).astype(float).ravel()
    s = np.asarray(s11_dB).astype(float).ravel()
    if f.size != s.size:
        raise ValueError("freq and s11_dB must have the same length")

    order = np.argsort(f)
    f = f[order]
    s = s[order]

    # Find global minimum (deepest dip)
    idx_min = int(np.nanargmin(s))
    f0 = f[idx_min]
    s_min = s[idx_min]

    # Helper: linear interpolate x at given y between two samples
    def interp_x_at_y(x0, y0, x1, y1, y_target):
        if y1 == y0:
            return (x0 + x1) / 2.0
        t = (y_target - y0) / (y1 - y0)
        return x0 + t * (x1 - x0)

    # Build mask where S11 is below threshold (more negative than level_db)
    mask = s <= level_db
    n = f.size

    f_low = None
    f_high = None

    if mask[idx_min]:
        # Walk left to find the last index still below threshold
        jL = idx_min
        while jL > 0 and mask[jL - 1]:
            jL -= 1
        # Crossing is between (jL-1, jL) if jL > 0; else it extends to start
        if jL > 0:
            f_low = interp_x_at_y(f[jL - 1], s[jL - 1], f[jL], s[jL], level_db)
        else:
            f_low = f[0]  # below threshold from the start (no crossing inside range)

        # Walk right to find the last index still below threshold
        jR = idx_min
        while jR < n - 1 and mask[jR + 1]:
            jR += 1
        # Crossing is between (jR, jR+1) if jR < n-1; else it extends to end
        if jR < n - 1:
            f_high = interp_x_at_y(f[jR], s[jR], f[jR + 1], s[jR + 1], level_db)
        else:
            f_high = f[-1]
    else:
        # Minimum is above threshold -> no -10 dB band
        f_low = None
        f_high = None

    # Compute BW metrics
    BW = None
    FBW = None
    Q = None
    if (f_low is not None) and (f_high is not None) and (f_high > f_low):
        BW = f_high - f_low
        FBW = BW / f0 if f0 > 0 else None
        Q = f0 / BW if BW > 0 else None

    # Plot
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(f / 1e9, s, "k-", linewidth=2, label="$S_{11}$")
    ax.grid(True)
    ax.set_ylabel("S-Parameter (dB)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_title(title)

    if annotate:
        # Horizontal threshold
        ax.axhline(level_db, linestyle="--", linewidth=1, alpha=0.7)

        # Minimum marker & vertical line
        ax.plot([f0 / 1e9], [s_min], "ro", label="min")
        ax.axvline(f0 / 1e9, color="r", linestyle=":", linewidth=1)

        # Band edges & shaded band
        if (f_low is not None) and (f_high is not None) and (f_high > f_low):
            ax.axvline(f_low / 1e9, color="C0", linestyle="--", linewidth=1)
            ax.axvline(f_high / 1e9, color="C0", linestyle="--", linewidth=1)
            ax.axvspan(f_low / 1e9, f_high / 1e9, alpha=0.12)

            # Labels
            txt = (f"$f_0 = {f0 / 1e9:.3f}\\,\\mathrm{{GHz}}\\; ({s_min:.1f}\\,\\mathrm{{dB}})$\n"
                   f"$f_{{-10\\,dB}}^\\mathrm{{low}} = {f_low / 1e9:.3f}\\,\\mathrm{{GHz}}$   "
                   f"$f_{{-10\\,dB}}^\\mathrm{{high}} = {f_high / 1e9:.3f}\\,\\mathrm{{GHz}}$\n"
                   f"$\\mathrm{{BW}} = {BW / 1e6:.1f}\\,\\mathrm{{MHz}}$   "
                   f"$\\mathrm{{FBW}} = {FBW * 100:.2f}\\%$   "
                   f"$Q \\approx {Q:.1f}$")
        else:
            # Only min is valid
            txt = (f"$f_0 = {f0 / 1e9:.3f}\\,\\mathrm{{GHz}}\\; ({s_min:.1f}\\,\\mathrm{{dB}})$\n"
                   f"No crossing at {level_db:.0f} dB")

        # Place text in a nice box
        ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

        # Legend (avoid duplicates)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    plt.tight_layout()

    return {
        "f0_Hz": f0,
        "S11_min_dB": s_min,
        "f_low_Hz": f_low,
        "f_high_Hz": f_high,
        "BW_Hz": BW,
        "FBW": FBW,
        "Q": Q,
    }


def coord_to_index(x=None, y=None, z=None, fd=None):
    """
    Convert physical coordinates (m) to nearest indices in E_fd array.
    Any of x, y, z may be None.
    """
    ix = (np.abs(fd.x - x)).argmin() if x is not None else None
    iy = (np.abs(fd.y - y)).argmin() if y is not None else None
    iz = (np.abs(fd.z - z)).argmin() if z is not None else None
    return ix, iy, iz


def phase_ref_from_baseline(y_m, E_line, halfwin=2):
    """
    y_m: 1D coords along the cut (meters)
    E_line: complex Ez phasor along the same y (shape [Ny])
    Returns (y_ref, phi_ref) taken at the amplitude maximum (window-averaged).
    """
    k = int(np.argmax(np.abs(E_line)))
    i0 = max(0, k - halfwin);
    i1 = min(len(E_line), k + halfwin + 1)
    # window-weighted phase near the max (reduces noise)
    w = np.abs(E_line[i0:i1])
    phi_ref = np.angle(np.sum(w * E_line[i0:i1]))
    y_ref = float(y_m[k])
    return y_ref, phi_ref


def align_line_to_reference(y_m, E_line, y_ref, phi_ref, halfwin=2):
    """
    Aligns E_line so that the phase at y_ref matches phi_ref.
    Returns E_aligned and the applied phase shift (radians).
    """
    # nearest index to the same physical y (you can replace with interpolation if you prefer)
    k = int(np.argmin(np.abs(y_m - y_ref)))
    i0 = max(0, k - halfwin);
    i1 = min(len(E_line), k + halfwin + 1)
    w = np.abs(E_line[i0:i1])
    # robust current phase near y_ref
    phi_cur = np.angle(np.sum(w * E_line[i0:i1])) if np.sum(w) > 0 else 0.0
    dphi = phi_cur - phi_ref
    return E_line * np.exp(-1j * dphi), dphi


def align_phase_global(E_line, E_line_ref, weight=None):
    """
    Align E_line to E_line_ref by a single complex scalar (LS optimum).
    Returns E_line_aligned, phi (radians).
    """
    if weight is None:
        weight = np.ones_like(E_line_ref, dtype=float)

    # mask out low-magnitude points in the reference to avoid noisy phase
    mask = np.abs(E_line_ref) > (0.05 * np.max(np.abs(E_line_ref)))
    w = weight * mask

    # complex scalar minimizing || E_ref - alpha * E ||
    # alpha* = (E_ref^H W E) / (E^H W E)
    num = np.vdot(E_line_ref[mask] * w[mask], E_line[mask])
    den = np.vdot(E_line[mask] * w[mask], E_line[mask]) + 1e-30
    alpha = num / den
    phi = np.angle(alpha)

    E_aligned = E_line * np.exp(-1j * phi)
    return E_aligned, phi


def find_max_ampl_phase(ref_line, amp_floor_frac=0.05, return_width=False):
    """
    Find phase dphi (radians) that maximizes ptp(Re(ref_line * exp(-1j*dphi))).

    Parameters
    ----------
    ref_line : (N,) complex array
        Complex field samples along a line.
    amp_floor_frac : float, default 0.05
        Ignore samples with |E| < amp_floor_frac * max(|E|) to reduce phase noise
        in low-amplitude regions. Set to 0 to disable.
    return_width : bool, default False
        If True, also return the achieved peak-to-peak width.

    Returns
    -------
    dphi : float
        Phase rotation to maximize min/max difference of the snapshot.
        Use: snapshot = np.real(ref_line * np.exp(-1j * dphi))
    width : float  (optional)
        The resulting max-min value of the snapshot (same units as |E|).
    """
    z = np.asarray(ref_line, dtype=np.complex128).ravel()

    # Keep finite samples only
    finite = np.isfinite(z)
    z = z[finite]
    if z.size == 0:
        return (0.0, 0.0) if return_width else 0.0

    # Optional amplitude floor to avoid outlier phases in tiny-|E| regions
    if amp_floor_frac > 0:
        m = np.abs(z)
        thr = amp_floor_frac * m.max()
        keep = m >= thr
        # Ensure we keep at least 2 points
        if keep.sum() >= 2:
            z = z[keep]
        elif z.size >= 2:
            idx = np.argsort(m)[-2:]
            z = z[np.sort(idx)]
        else:
            # Only one point survives: align to its own phase
            dphi = float(np.angle(z[0]))
            return (dphi, 0.0) if return_width else dphi

    if z.size == 1:
        dphi = float(np.angle(z[0]))
        return (dphi, 0.0) if return_width else dphi

    # Find the farthest pair (diameter) without allocating an NxN matrix
    max_d2 = -1.0
    imax = jmax = 0
    for i in range(1, z.size):
        w = z[i] - z[:i]  # vectorized diffs to previous points
        d2 = (w.real * w.real) + (w.imag * w.imag)
        j = int(np.argmax(d2))
        d2max = float(d2[j])
        if d2max > max_d2:
            max_d2 = d2max
            imax, jmax = i, j

    w = z[imax] - z[jmax]
    dphi = float(np.angle(w))  # project along the diameter direction
    width = float(np.sqrt(max_d2))  # equals max-min after applying dphi

    return (dphi, width) if return_width else dphi


from typing import Union

ArrayLike = Union[np.ndarray, float, complex]


def s2abcd(s11: ArrayLike, s21: ArrayLike, s12: ArrayLike, s22: ArrayLike,
           zref: ArrayLike = 50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert 2-port S-parameters (equal reference impedance zref) to ABCD.

    Parameters
    ----------
    s11, s21, s12, s22 : arrays of shape (N,) (complex)
    zref : scalar or array-like (ohms), same broadcastable shape

    Returns
    -------
    A, B, C, D : complex arrays of shape (N,)
    """
    s11 = np.asarray(s11, dtype=complex)
    s21 = np.asarray(s21, dtype=complex)
    s12 = np.asarray(s12, dtype=complex)
    s22 = np.asarray(s22, dtype=complex)
    zref = np.asarray(zref, dtype=complex)

    # Avoid division by zero (where S21 ~ 0); mark those as NaN
    tiny = 1e-14
    bad = np.abs(s21) < tiny
    s21_safe = np.where(bad, np.nan + 1j * np.nan, s21)

    A = ((1 + s11) * (1 - s22) + s12 * s21_safe) / (2 * s21_safe)
    B = zref * ((1 + s11) * (1 + s22) - s12 * s21_safe) / (2 * s21_safe)
    C = (1 / zref) * ((1 - s11) * (1 - s22) - s12 * s21_safe) / (2 * s21_safe)
    D = ((1 - s11) * (1 + s22) + s12 * s21_safe) / (2 * s21_safe)

    # Propagate NaNs to the same indices
    A[bad] = np.nan
    B[bad] = np.nan
    C[bad] = np.nan
    D[bad] = np.nan
    return A, B, C, D


def z0_from_abcd(B: ArrayLike, C: ArrayLike) -> np.ndarray:
    """
    Characteristic impedance from ABCD of a uniform line: Z0 = sqrt(B/C)
    with branch selection so Re{Z0} >= 0.
    """
    B = np.asarray(B, dtype=complex)
    C = np.asarray(C, dtype=complex)
    Z0 = np.sqrt(B / C)

    # Choose the sign branch: make real part non-negative; if ~0, make imag >= 0
    flip = (np.real(Z0) < 0) | ((np.abs(np.real(Z0)) < 1e-12) & (np.imag(Z0) < 0))
    Z0[flip] = -Z0[flip]
    return Z0


def z0_from_s(s11: ArrayLike, s21: ArrayLike, s12: ArrayLike, s22: ArrayLike,
              zref: ArrayLike = 50.0) -> np.ndarray:
    """Convenience wrapper: S -> ABCD -> Z0."""
    A, B, C, D = s2abcd(s11, s21, s12, s22, zref=zref)
    return z0_from_abcd(B, C)


def write_s2p_file(name, freq, Z0, s11, s21, s12, s22):
    S = np.zeros((len(freq), 2, 2), dtype=complex)
    S[:, 0, 0] = s11
    S[:, 1, 1] = s22
    S[:, 1, 0] = s21
    S[:, 0, 1] = s12

    ntw = rf.Network(frequency=freq, s=S, z0=Z0)
    ntw.write_touchstone(name)

def write_s1p_file(name, freq, Z0, s11):
    S = np.zeros((len(freq), 1, 1), dtype=complex)
    S[:, 0, 0] = s11

    ntw = rf.Network(frequency=freq, s=S, z0=Z0)
    ntw.write_touchstone(name)
