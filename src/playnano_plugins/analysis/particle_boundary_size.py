"""
Particle boundary size analysis module for the playNano analysis pipeline.

This module computes a per-particle boundary size metric over time using
labeled region masks produced by detection modules such as
``feature_detection`` and particle associations from ``particle_tracking``.

For each tracked particle and frame, the maximum dimension of the particle's
bounding box is measured. Optionally, a threshold may be applied to classify
each time point into discrete states (e.g. compact vs extended).

The module is designed to integrate with the playNano ``AnalysisPipeline`` and
records results in a provenance-aware, serializable format suitable for
downstream analysis and plotting.
"""

import warnings
from typing import Any, Optional

import numpy as np
from playnano.analysis.base import AnalysisModule
from skimage.measure import label as sk_label, regionprops


def _bbox_max_dim_from_binary_mask(binary_mask: np.ndarray) -> float:
    """
    Compute the maximum bounding-box dimension of a binary object mask.

    The function identifies connected components in the input mask, selects
    the largest component by area, and returns the maximum of the bounding-box
    height and width for that component.

    Parameters
    ----------
    binary_mask : ndarray of bool
        Boolean array where ``True`` indicates pixels belonging to the object
        of interest.

    Returns
    -------
    float
        Maximum of the bounding-box height and width for the largest connected
        component. Returns ``np.nan`` if no foreground pixels are present.
    """
    if binary_mask.dtype != bool:
        binary_mask = binary_mask.astype(bool)

    if not np.any(binary_mask):
        return np.nan

    labeled = sk_label(binary_mask)
    props = regionprops(labeled)
    if not props:
        return np.nan

    region = max(props, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox
    return float(max(maxr - minr, maxc - minc))


class BoundarySizeModule(AnalysisModule):
    """
    Measure particle boundary size over time from tracked, labeled detections.

    This analysis module computes a per-frame boundary size metric for each
    tracked particle by extracting the labeled region associated with that
    particle and measuring the maximum bounding-box dimension.

    The module requires:
    - output from a particle tracking module (e.g. ``particle_tracking``)
    - output from a labeled detection module providing:
        * ``features_per_frame`` with per-feature labels
        * ``labeled_masks`` for each frame

    Optionally, a threshold can be applied to derive a discrete state variable
    from the boundary size (e.g. compact vs extended).

    Notes
    -----
    - Out-of-range frame or point indices are handled gracefully by inserting
      ``np.nan`` values and issuing warnings.
    - If a feature does not provide a label, a ``RuntimeError`` is raised, as
      labeled regions are required for this analysis.
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """
        Name used to register this module in the analysis pipeline.

        Returns
        -------
        str
            Module name.
        """
        return "particle_boundary_size"

    # Pipeline 'requires' check is any-of; explicit validation is done in run().
    requires = ["particle_tracking"]

    def run(
        self,
        stack,
        previous_results: Optional[dict[str, Any]] = None,
        *,
        tracking_module: str = "particle_tracking",
        detection_module: str = "feature_detection",
        threshold: Optional[float] = None,
        measure: str = "bbox_max_dim",
        label_key: str = "label",
    ) -> dict[str, Any]:
        """
        Execute the boundary size analysis.

        Parameters
        ----------
        stack : AFMImageStack
            Input image stack on which the analysis is performed.
        previous_results : dict, optional
            Dictionary of results from previously executed analysis modules.
            Must include outputs from the specified tracking and detection
            modules.
        tracking_module : str, default="particle_tracking"
            Name of the analysis module providing particle tracking results.
        detection_module : str, default="feature_detection"
            Name of the analysis module providing labeled detection results.
        threshold : float or None, optional
            Threshold applied to the boundary size metric to compute a binary
            state variable. If ``None``, no state classification is performed.
        measure : str, default="bbox_max_dim"
            Boundary size metric to compute. Currently only
            ``"bbox_max_dim"`` is supported.
        label_key : str, default="label"
            Key used to access region labels in the per-feature dictionaries.

        Returns
        -------
        dict
            Analysis output dictionary with the following keys:

            ``measure`` : str
                Name of the boundary size metric.
            ``threshold`` : float or None
                Threshold used for state classification.
            ``per_track`` : list of dict
                Per-track time series of frames, timestamps, boundary sizes,
                and optional state values.
            ``flat_table`` : list of dict
                Flattened, row-wise representation suitable for conversion to
                a pandas DataFrame.
            ``plot_hints`` : dict
                Suggested plotting helper functions for visualizing results.
            ``summary`` : dict
                Summary statistics and bookkeeping information.
        """
        if previous_results is None:
            raise RuntimeError(f"{self.name!r} requires previous results to run.")

        if measure != "bbox_max_dim":
            raise ValueError(
                f"{self.name!r} only supports measure='bbox_max_dim' right now."
            )

        if tracking_module not in previous_results:
            raise RuntimeError(
                f"{self.name!r} requires tracking_module={tracking_module!r} "
                f"to be present in previous_results."
            )

        track_out = previous_results[tracking_module]
        if "tracks" not in track_out:
            raise RuntimeError(
                f"{self.name!r} expected '{tracking_module}' output to contain 'tracks'."  # noqa
            )

        if detection_module not in previous_results:
            raise RuntimeError(
                f"{self.name!r} requires detection_module={detection_module!r} "
                f"to be present in previous_results."
            )

        det_out = previous_results[detection_module]
        if "labeled_masks" not in det_out or "features_per_frame" not in det_out:
            raise RuntimeError(
                f"{self.name!r} requires detection output with 'labeled_masks' and "
                f"'features_per_frame' (e.g. feature_detection)."
            )

        labeled_masks = det_out["labeled_masks"]
        features_per_frame = det_out["features_per_frame"]

        n_frames = min(len(labeled_masks), len(features_per_frame))
        if n_frames == 0:
            return {
                "measure": measure,
                "threshold": threshold,
                "per_track": [],
                "flat_table": [],
                "summary": {
                    "n_tracks": 0,
                    "n_rows": 0,
                    "n_skipped_index_errors": 0,
                    "n_missing_region_measurements": 0,
                    "state_included": threshold is not None,
                },
            }

        rows = []
        per_track = []
        n_skipped = 0
        n_missing = 0

        for trk in track_out["tracks"]:
            track_id = int(trk["id"])
            frames = list(trk.get("frames", []))
            pt_indices = list(trk.get("point_indices", []))

            track_frames = []
            track_timestamps = []
            track_max_dim = []
            track_state = [] if threshold is not None else None

            if len(frames) != len(pt_indices):
                warnings.warn(
                    f"[{self.name}] track_id={track_id}: frames length ({len(frames)}) != "  # noqa
                    f"point_indices length ({len(pt_indices)}). Truncating to shortest.",  # noqa
                    stacklevel=2,
                )

            for frame_idx, pt_idx in zip(frames, pt_indices, strict=True):
                frame_idx = int(frame_idx)
                pt_idx = int(pt_idx)

                # Timestamp
                try:
                    ts = float(stack.time_for_frame(frame_idx))
                except Exception:
                    ts = float(frame_idx)

                label_val = np.nan  # default when missing/out-of-range
                max_dim = np.nan

                # Frame bounds
                if frame_idx < 0 or frame_idx >= n_frames:
                    warnings.warn(
                        f"[{self.name}] track_id={track_id}: frame {frame_idx} out of range. Writing NaN.",  # noqa
                        stacklevel=2,
                    )
                    n_skipped += 1
                else:
                    feats_this = features_per_frame[frame_idx]
                    if pt_idx < 0 or pt_idx >= len(feats_this):
                        warnings.warn(
                            f"[{self.name}] track_id={track_id}, frame={frame_idx}: point_index {pt_idx} out of range. Writing NaN.",  # noqa
                            stacklevel=2,
                        )
                        n_skipped += 1
                    else:
                        feat = feats_this[pt_idx]
                        if label_key not in feat or feat[label_key] is None:
                            raise RuntimeError(
                                f"[{self.name}] Feature dict at frame={frame_idx}, point_index={pt_idx} "  # noqa
                                f"does not contain '{label_key}'. This module requires labeled regions "  # noqa
                                f"(e.g. from feature_detection)."
                            )

                        label_val = int(feat[label_key])
                        lm = np.asarray(labeled_masks[frame_idx])
                        obj_mask = lm == label_val
                        max_dim = _bbox_max_dim_from_binary_mask(obj_mask)

                        if np.isnan(max_dim):
                            n_missing += 1

                track_frames.append(frame_idx)
                track_timestamps.append(ts)
                track_max_dim.append(float(max_dim))

                row = {
                    "track_id": track_id,
                    "label": (int(label_val) if not np.isnan(label_val) else np.nan),
                    "frame": frame_idx,
                    "timestamp": ts,
                    "max_dim": float(max_dim) if not np.isnan(max_dim) else np.nan,
                }

                if threshold is not None:
                    if np.isnan(max_dim):
                        state_val = np.nan
                    else:
                        state_val = int(max_dim > threshold)

                    track_state.append(state_val)
                    row["state"] = state_val

                rows.append(row)

            trk_rec = {
                "track_id": track_id,
                "frames": track_frames,
                "timestamps": track_timestamps,
                "max_dim": track_max_dim,
            }
            if threshold is not None:
                trk_rec["state"] = track_state

            per_track.append(trk_rec)

        return {
            "measure": measure,
            "threshold": threshold,
            "per_track": per_track,
            "flat_table": rows,
            "plot_hints": {
                "single_track": "playnano_plugins.plotting.plot_boundary_over_time",
                "multi_track": "playnano_plugins.plotting.plot_boundary_over_time_multiple_tracks",  # noqa
            },
            "summary": {
                "n_tracks": len(track_out["tracks"]),
                "n_rows": len(rows),
                "n_skipped_index_errors": n_skipped,
                "n_missing_region_measurements": n_missing,
                "state_included": threshold is not None,
            },
        }
