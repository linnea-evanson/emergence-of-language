"""Brain plots of the paper: electrodes on a lateral view of the fsaverage cortex.

The fsaverage surfaces come from MNE
(``mne.datasets.fetch_fsaverage``, downloaded once to ``~/mne_data``).

Both hemispheres are drawn on the left lateral view (y, z coordinates); electrodes
of the medial wall (``data.electrodes().medial``) are not shown on it.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BACKGROUND_GRAY = (0.7, 0.7, 0.7)
CMAP_LIGHT_GRAY = LinearSegmentedColormap.from_list("light_gray", [(0.2, 0.2, 0.2), (1, 1, 1)], N=256)
#: Above this number of points a scatter is rasterized in vector outputs.
RASTER_POINTS = 5000


@functools.cache
def fsaverage_dir() -> Path:
    import mne

    return Path(mne.datasets.fetch_fsaverage(verbose=False))


@functools.cache
def pial_surfaces() -> tuple[np.ndarray, np.ndarray]:
    """Left and right fsaverage pial vertices, in metres."""
    import mne

    return tuple(  # type: ignore[return-value]
        mne.read_surface(str(fsaverage_dir() / "surf" / f"{h}.pial"))[0] / 1e3 for h in ("lh", "rh"))


@functools.cache
def curvatures() -> tuple[np.ndarray, np.ndarray]:
    """Left and right (sign-flipped) fsaverage curvature, for the cortex shading."""
    from mne.surface import read_curvature

    return tuple(  # type: ignore[return-value]
        -read_curvature(str(fsaverage_dir() / "surf" / f"{h}.curv"), binary=False) for h in ("lh", "rh"))


def brain_background(ax, hemi: str = "left", shading: bool = True) -> None:
    """Lateral view of one hemisphere: grey silhouette, shaded by curvature."""
    surface = pial_surfaces()[0 if hemi == "left" else 1]
    curv = curvatures()[0 if hemi == "left" else 1]
    order = np.argsort(surface[:, 0])
    if hemi == "left":
        order = order[::-1]  # draw from medial to lateral
    sign = -1 if hemi == "left" else 1
    ax.scatter(sign * surface[order, 1], surface[order, 2], color=BACKGROUND_GRAY, zorder=-100000,
               rasterized=True)
    if shading:
        ax.scatter(sign * surface[order, 1], surface[order, 2], c=curv[order], cmap=CMAP_LIGHT_GRAY,
                   s=0.5, rasterized=True)
    ax.set_axis_off()


def get_sizes(vals: np.ndarray, vmin: float, vmax: float, smax: float = 100, s0: float = 10) -> np.ndarray:
    """Marker sizes growing with the (clipped) value, from ``s0``."""
    s = np.clip(vals, vmin, vmax)
    return smax * (s - vmin) / (vmax - vmin) + s0


def _label(v: float) -> str:
    return str(round(v, 2)).replace("0.", ".")


def plot_legend(ax, vsmall: float, scatter, vmin: float, vmax: float, fs: float = 14, s0: float = 10,
                title: str = "", plot_ns: bool = True, anchor: tuple[float, float] = (0, 0)) -> None:
    for s, v in [(102, vmax), (50, (vmin + vmax) / 2), (s0, vsmall)]:
        ax.scatter([], [], s=s, color=scatter.to_rgba(v), label=_label(v), edgecolors="k", linewidth=0.5)
    if plot_ns:
        ax.scatter([], [], s=2, color="k", label="n.s.", edgecolors="k", linewidth=0.5)
    ax.legend(title=title, loc="lower left", bbox_to_anchor=anchor, fontsize=fs, title_fontsize=fs,
              borderaxespad=0, borderpad=0.2, frameon=False)


def plot_brain_values(ax, values: pd.DataFrame, selected: set[tuple[str, str]] | None = None,
                      col: str = "score", vmax: float = 0.5, vmin: float = 0.0, cmap="Blues_r",
                      title: str = r"$\mathcal{R}$", fs: float = 10, legend: bool = True,
                      anchor: tuple[float, float] = (0.75, -0.12), smax: float = 100,
                      shading: bool = True) -> pd.DataFrame:
    """Electrode values on the left lateral cortex.

    ``values``: participant, electrode, ``col``. Electrodes without a position or on the
    medial wall are skipped. ``selected`` (participant, electrode) are drawn coloured and
    sized by value; the others as small black dots (``None``: all selected). Returns the
    electrodes drawn, with their position.
    """
    from . import data

    el = data.electrodes()
    el = el[el.x.notna() & ~el.medial][["participant_id", "electrode_id", "y", "z"]]
    df = values.merge(el, left_on=["participant", "electrode"],
                      right_on=["participant_id", "electrode_id"]).drop(
        columns=["participant_id", "electrode_id"])
    sel = (np.ones(len(df), bool) if selected is None else
           np.array([k in selected for k in zip(df.participant, df.electrode)], bool))
    df = df.assign(selected=sel, s=get_sizes(df[col].to_numpy(), vmin, vmax, smax=smax))
    df = df.sort_values("s", kind="stable")

    brain_background(ax, shading=shading)
    ns = df[~df.selected]
    ax.scatter(-ns.y, ns.z, color="k", s=2, edgecolors="k", linewidth=0.5)
    sig = df[df.selected]
    scatter = ax.scatter(-sig.y, sig.z, c=sig[col], s=sig.s, edgecolors="k", linewidth=0.5,
                         cmap=cmap, vmin=vmin, vmax=vmax)
    if legend:
        pos = sig[col][sig[col] > 0]
        vsmall = float(pos.min()) if len(pos) else 0.01
        plot_legend(ax, vsmall, scatter, vmin, vmax, fs=fs, title=title, anchor=anchor,
                    plot_ns=selected is not None)
    ax.set_axis_off()
    return df


def rasterize_large(fig, threshold: int = RASTER_POINTS) -> None:
    """Rasterize every scatter with more than ``threshold`` points (the cortex)."""
    for ax in fig.get_axes():
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if offsets is not None and len(offsets) > threshold:
                coll.set_rasterized(True)


def roi_inset(ax, s_all: float = 0.3, s_roi: float = 0.6) -> None:
    """Cortex with the located electrodes, those of each figure ROI in its colour."""
    from . import data
    from .style import ROI_COLORS

    brain_background(ax)
    el = data.electrodes()
    el = el[el.x.notna() & ~el.medial]
    ax.scatter(-el.y, el.z, s=s_all, color="0.65", zorder=1, rasterized=True)
    for roi in data.ROIS[1:]:
        sub = el[el.roi == roi]
        c = ROI_COLORS[roi][2 if roi == "IPL" else 1]
        ax.scatter(-sub.y, sub.z, s=s_roi, color=c, zorder=2, rasterized=True)
    ax.set_aspect("equal")
    ax.set_axis_off()

