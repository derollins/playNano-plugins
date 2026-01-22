# mypy: disable-error-code=type-arg
"""Plotting functions for particle boundary size analysis."""

from typing import Any, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from skimage.measure import label as sk_label, regionprops


def plot_boundary_over_time(
    boundary_out: dict[str, Any],
    *,
    track_id: Optional[int] = None,
    threshold: Optional[float] = None,
    x: str = "frame",
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """
    Plot max_dim and (optionally) state over time for one track.

    Parameters
    ----------
    boundary_out:
        Output dict from particle_boundary_size module (record["analysis"][...]).
    track_id:
        Track to plot. Defaults to first track.
    threshold:
        Overrides boundary_out["threshold"]. If None, uses value in output (if present).
    x:
        "frame" or "timestamp" for x-axis.
    """
    per_track = boundary_out.get("per_track", [])
    if not per_track:
        raise ValueError("No tracks found in boundary_out['per_track'].")

    if track_id is None:
        tr = per_track[0]
    else:
        matches = [t for t in per_track if int(t["track_id"]) == int(track_id)]
        if not matches:
            raise ValueError(f"track_id={track_id} not found.")
        tr = matches[0]

    frames = np.asarray(tr["frames"])
    times = np.asarray(tr["timestamps"], dtype=float)
    max_dim = np.asarray(tr["max_dim"], dtype=float)

    xvals = times if x == "timestamp" else frames
    xlabel = "Time (s)" if x == "timestamp" else "Frame"

    if threshold is None:
        threshold = boundary_out.get("threshold", None)

    # No threshold => single plot
    if threshold is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xvals, max_dim, label="max_dim")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Max bbox dimension (px)")
        ax.set_title(f"Boundary size over time (track_id={tr['track_id']})")
        ax.legend()
        fig.tight_layout()
        return fig

    # state (use computed if present; otherwise recompute)
    state = tr.get("state", None)
    if state is None:
        state = np.where(np.isnan(max_dim), np.nan, (max_dim > threshold).astype(int))
    else:
        state = np.asarray(state, dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(xvals, max_dim, label="max_dim")
    ax1.axhline(threshold, color="red", linestyle="--", label=f"threshold={threshold}")
    ax1.set_ylabel("Max bbox dimension (px)")
    ax1.set_title(f"Boundary size over time (track_id={tr['track_id']})")
    ax1.legend()

    ax2.step(xvals, state, where="mid")
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Compact", "Extended"])
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("State")

    fig.tight_layout()
    return fig


def plot_boundary_over_time_multiple_tracks(
    boundary_out: dict,
    *,
    track_ids: Optional[Iterable[int]] = None,
    max_tracks: int = 10,
    x: str = "frame",  # "frame" or "timestamp"
    figsize: Tuple[float, float] = (10, 5),
) -> Figure:
    """
    Plot max_dim over time for multiple tracks on a single axis.

    Parameters
    ----------
    boundary_out
        Output dict from particle_boundary_size module.
    track_ids
        Iterable of track IDs to plot. If None, the first `max_tracks` are used.
    max_tracks
        Maximum number of tracks to plot if track_ids is None.
    x
        X-axis variable: "frame" or "timestamp".
    figsize
        Figure size.
    """
    if "flat_table" not in boundary_out:
        raise ValueError("boundary_out must contain 'flat_table'.")

    df = pd.DataFrame(boundary_out["flat_table"])
    if df.empty:
        raise ValueError("boundary_out['flat_table'] is empty.")

    if x not in {"frame", "timestamp"}:
        raise ValueError("x must be 'frame' or 'timestamp'.")

    # Determine which tracks to plot
    if track_ids is None:
        track_ids_list: list[int] = (
            df["track_id"].dropna().unique().astype(int)[:max_tracks].tolist()
        )
    else:
        track_ids_list = list(track_ids)

    fig, ax = plt.subplots(figsize=figsize)

    for tid in track_ids_list:
        dft = df[df["track_id"] == tid].sort_values(x)
        if dft.empty:
            continue

        ax.plot(
            dft[x],
            dft["max_dim"],
            label=f"track {tid}",
        )

    xlabel = "Time (s)" if x == "timestamp" else "Frame"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Max bbox dimension")
    ax.set_title("Boundary size over time (multiple tracks)")

    if len(track_ids_list) <= 12:
        ax.legend(ncols=2, fontsize=8)
    else:
        ax.legend(fontsize=7, ncols=3)

    fig.tight_layout()
    return fig


def animate_boundary_size_crop(
    boundary_out: dict[str, Any],
    masks: Sequence[np.ndarray],
    *,
    track_id: Optional[int] = None,
    crop_size: int = 100,
    fps: int = 6,
    origin_lower: bool = True,
    threshold: Optional[float] = None,
    label_value_fallback: int = 1,
    figsize: Tuple[float, float] = (6, 6),
    save_path: Optional[str] = None,
) -> tuple[Figure, FuncAnimation]:
    """
    Animate a cropped mask region around a tracked particle over time.

    This is intended as a sanity-check visualisation for the
    ``particle_boundary_size`` analysis output. It crops a fixed-size window
    around the particle's region (using the labeled mask and the particle label),
    and colours the mask depending on whether the measured max bounding-box
    dimension exceeds a threshold.

    Parameters
    ----------
    boundary_out
        Output dictionary from the ``particle_boundary_size`` analysis module.
        Must contain ``per_track`` and/or ``flat_table`` with ``track_id`` and
        ``frame`` entries. If the analysis was run with threshold enabled,
        it may also contain ``threshold`` and/or per-row ``state``.
    masks
        Sequence of labeled masks per frame (e.g. feature_detection
        ``labeled_masks``). Each mask is a 2D integer array where region labels
        identify detected objects.
    track_id
        Track to animate. If None, the first track in ``boundary_out['per_track']``
        is used.
    crop_size
        Size of the square crop in pixels.
    fps
        Frames per second for the animation.
    origin_lower
        Whether to display images with origin="lower" (typical for image coords).
    threshold
        Threshold used to colour the mask. If None, uses
        ``boundary_out.get("threshold")``.
    label_value_fallback
        If the expected label is not present in a frame (rare), this value is used
        for cropping (default 1). You may prefer to set this to 0 to show blank.
    figsize
        Matplotlib figure size.
    save_path
        Optional path to save the animation. Supported: ``.gif``, ``.mp4``, ``.m4v``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    anim : matplotlib.animation.FuncAnimation
        The animation object.

    Notes
    -----
    This function assumes your labeled masks are aligned with the tracking output
    (i.e. frame indices match and labels correspond to detected regions).
    """
    if threshold is None:
        threshold = boundary_out.get("threshold", None)

    # Build a DataFrame from flat_table (easy filtering by frame/track)
    if "flat_table" not in boundary_out:
        raise ValueError("boundary_out must contain 'flat_table'.")

    df = pd.DataFrame(boundary_out["flat_table"])
    if df.empty:
        raise ValueError("boundary_out['flat_table'] is empty.")

    if track_id is None:
        # Prefer per_track if available (stable ordering), else first track_id in df
        if boundary_out.get("per_track"):
            track_id = int(boundary_out["per_track"][0]["track_id"])
        else:
            track_id = int(df["track_id"].dropna().unique()[0])

    dft = df[df["track_id"] == track_id].sort_values("frame")
    if dft.empty:
        raise ValueError(f"No rows found for track_id={track_id} in flat_table.")

    frames = dft["frame"].astype(int).to_list()

    # --- helpers -------------------------------------------------------------
    def _crop(arr: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
        half = size // 2
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half

        sub = arr[
            max(y1, 0) : min(y2, arr.shape[0]), max(x1, 0) : min(x2, arr.shape[1])
        ]

        pad_top = max(0, 0 - y1)
        pad_bottom = max(0, y2 - arr.shape[0])
        pad_left = max(0, 0 - x1)
        pad_right = max(0, x2 - arr.shape[1])

        sub = np.pad(
            sub,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        return sub[:size, :size]

    def _centroid_from_binary_mask(binary_mask: np.ndarray) -> tuple[int, int] | None:
        """Return (cx, cy) centroid of largest component in binary_mask, or None."""
        if not np.any(binary_mask):
            return None
        labeled = sk_label(binary_mask.astype(bool))
        props = regionprops(labeled)
        if not props:
            return None
        region = max(props, key=lambda r: r.area)
        cy, cx = region.centroid  # (row, col)
        return int(cx), int(cy)

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        np.zeros((crop_size, crop_size)),
        cmap="afmhot",
        origin="lower" if origin_lower else "upper",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xlabel("x (cropped)")
    ax.set_ylabel("y (cropped)")

    # --- animation update ----------------------------------------------------
    def update(i: int) -> list[Any]:
        frame_idx = frames[i]
        if frame_idx < 0 or frame_idx >= len(masks):
            ax.set_title(f"Frame {frame_idx} out of range")
            im.set_data(np.zeros((crop_size, crop_size)))
            return [im]

        lm = np.asarray(masks[frame_idx])

        row = dft[dft["frame"] == frame_idx]
        if row.empty:
            # Shouldn't happen, but be robust
            im.set_data(np.zeros((crop_size, crop_size)))
            ax.set_title(f"Track {track_id} | Frame {frame_idx} | no data")
            return [im]

        max_dim_val = (
            float(row["max_dim"].iloc[0]) if "max_dim" in row.columns else np.nan
        )

        # label may be NaN for out-of-range pt_idx/frame; handle gracefully
        label_val = row["label"].iloc[0] if "label" in row.columns else np.nan
        if label_val is None or (isinstance(label_val, float) and np.isnan(label_val)):
            crop_img = np.zeros((crop_size, crop_size))
            ax.set_title(
                f"Track {track_id} | Frame {frame_idx} | label missing | max_dim={max_dim_val:.2f}"  # noqa
            )
        else:
            label_val = int(label_val)

            # Mask for the tracked object
            obj_mask = lm == label_val

            # centroid from the tracked object (largest component only)
            c = _centroid_from_binary_mask(obj_mask)
            if c is None:
                crop_img = np.zeros((crop_size, crop_size))
                ax.set_title(
                    f"Track {track_id} | Frame {frame_idx} | label={label_val} missing | max_dim={max_dim_val:.2f}"  # noqa
                )
            else:
                cx, cy = c
                crop_img = _crop((lm == label_val).astype(np.uint8), cx, cy, crop_size)

                ax.set_title(
                    f"Track {track_id} | Frame {frame_idx} | label={label_val} | max_dim={max_dim_val:.2f}"  # noqa
                )

        # Colour switch based on threshold (if provided)
        if threshold is not None and not np.isnan(max_dim_val):
            im.set_cmap("inferno" if max_dim_val > threshold else "afmhot")
        else:
            im.set_cmap("afmhot")

        im.set_data(crop_img)
        im.set_clim(vmin=0, vmax=1)  # binary image

        ax.set_xlim(0, crop_size)
        ax.set_ylim(0, crop_size)
        if not origin_lower:
            ax.set_ylim(crop_size, 0)

        return [im]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=True,
    )

    if save_path:
        ext = save_path.lower().rsplit(".", 1)[-1]
        if ext == "gif":
            anim.save(save_path, writer="pillow", fps=fps)
        elif ext in ("mp4", "m4v"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save_path must end with .gif or .mp4/.m4v")

    return fig, anim
