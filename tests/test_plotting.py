import matplotlib
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from playnano_plugins.plotting.particle_boundary_size import (
    animate_boundary_size_crop,
    plot_boundary_over_time,
    plot_boundary_over_time_multiple_tracks,
)

matplotlib.use("Agg")


def _fake_boundary_out(with_threshold: bool = True):
    # minimal structure to satisfy plotting funcs
    per_track = [
        {
            "track_id": 0,
            "frames": [0, 1, 2],
            "timestamps": [0.0, 0.1, 0.2],
            "max_dim": [10.0, 20.0, 30.0],
        }
    ]
    if with_threshold:
        per_track[0]["state"] = [0, 0, 1]

    flat_table = [
        {
            "track_id": 0,
            "label": 1,
            "frame": 0,
            "timestamp": 0.0,
            "max_dim": 10.0,
            "state": 0,
        },
        {
            "track_id": 0,
            "label": 1,
            "frame": 1,
            "timestamp": 0.1,
            "max_dim": 20.0,
            "state": 0,
        },
        {
            "track_id": 0,
            "label": 1,
            "frame": 2,
            "timestamp": 0.2,
            "max_dim": 30.0,
            "state": 1,
        },
    ]

    out = {
        "per_track": per_track,
        "flat_table": flat_table,
        "threshold": 25.0 if with_threshold else None,
    }
    return out


def _fake_masks(n_frames=3, shape=(200, 200), label_val=1):
    masks = []
    for i in range(n_frames):
        lm = np.zeros(shape, dtype=int)
        # simple square blob
        lm[50 + i : 60 + i, 80:90] = label_val
        masks.append(lm)
    return masks


def test_plot_boundary_over_time_returns_figure_with_threshold():
    out = _fake_boundary_out(with_threshold=True)
    fig = plot_boundary_over_time(out, track_id=0, x="frame")
    assert isinstance(fig, Figure)


def test_plot_boundary_over_time_returns_figure_without_threshold():
    out = _fake_boundary_out(with_threshold=False)
    # remove threshold entirely to force the single-panel branch
    out.pop("threshold", None)
    # also remove state to force recompute / no-state path
    out["per_track"][0].pop("state", None)

    fig = plot_boundary_over_time(out, track_id=0, x="timestamp", threshold=None)
    assert isinstance(fig, Figure)


def test_plot_boundary_over_time_raises_if_no_per_track():
    with pytest.raises(ValueError, match="No tracks found"):
        plot_boundary_over_time({"per_track": []})


def test_plot_boundary_over_time_multiple_tracks_returns_figure():
    out = _fake_boundary_out(with_threshold=True)
    fig = plot_boundary_over_time_multiple_tracks(out, max_tracks=5, x="frame")
    assert isinstance(fig, Figure)


def test_plot_boundary_over_time_multiple_tracks_requires_flat_table():
    out = _fake_boundary_out(with_threshold=True)
    out.pop("flat_table")
    with pytest.raises(ValueError, match="flat_table"):
        plot_boundary_over_time_multiple_tracks(out)


def test_animate_boundary_size_crop_returns_animation():
    out = _fake_boundary_out(with_threshold=True)
    masks = _fake_masks(n_frames=3)

    fig, anim = animate_boundary_size_crop(out, masks, track_id=0, crop_size=50, fps=2)
    assert isinstance(fig, Figure)
    assert isinstance(anim, FuncAnimation)


def test_animate_boundary_size_crop_handles_missing_label_gracefully():
    out = _fake_boundary_out(with_threshold=True)
    # Set label to NaN for one frame (simulates out-of-range pt_idx)
    out["flat_table"][1]["label"] = np.nan

    masks = _fake_masks(n_frames=3)
    fig, anim = animate_boundary_size_crop(out, masks, track_id=0, crop_size=50, fps=2)

    assert isinstance(fig, Figure)
    assert isinstance(anim, FuncAnimation)
