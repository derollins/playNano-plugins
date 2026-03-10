# mypy: disable-error-code=type-arg

"""
playNano plugin module to apply Topostats filtering operations within playNano.

This ignores most of the Topostats internals, generates a minimal ``TopoStats`` object
with an input 2D numpy array ``frame`` which is augmented by ``pixel_to_nm_scaling``
(default value: 1) and ``filename`` with value "frame".
"""

import logging
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _deep_update(
    dst: MutableMapping[str, Any],
    src: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Recursively update mapping ``dst`` with values from ``src``.

    Nested mappings are merged; non-mapping values overwrite existing entries.
    """
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), MutableMapping):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _default_filter_config() -> dict[str, Any]:
    # Aligned with the filter section of the TopoStats config file.
    return {
        "row_alignment_quantile": 0.5,
        "threshold_method": "std_dev",  # "otsu" | "std_dev" | "absolute"
        "otsu_threshold_multiplier": 1.0,
        "threshold_std_dev": {"below": 10.0, "above": 1.0},
        "threshold_absolute": {"below": -1.0, "above": 1.0},
        "gaussian_size": 1.0121397464510862,
        "gaussian_mode": "nearest",
        "remove_scars": {
            "run": False,
            "removal_iterations": 2,
            "threshold_low": 0.250,
            "threshold_high": 0.666,
            "max_scar_width": 4,
            "min_scar_length": 16,
        },
    }


def _build_filter_config(
    *,
    filter_config: Optional[Mapping[str, Any]] = None,
    # ---- explicit CLI/API knobs (top-level) ----
    row_alignment_quantile: Optional[float] = None,
    threshold_method: Optional[str] = None,
    otsu_threshold_multiplier: Optional[float] = None,
    gaussian_size: Optional[float] = None,
    gaussian_mode: Optional[str] = None,
    # ---- std-dev thresholds ----
    threshold_std_dev_above: Optional[float] = None,
    threshold_std_dev_below: Optional[float] = None,
    # ---- absolute thresholds ----
    threshold_absolute_above: Optional[float] = None,
    threshold_absolute_below: Optional[float] = None,
    # ---- scar removal knobs ----
    remove_scars: Optional[bool] = None,
    scars_removal_iterations: Optional[int] = None,
    scars_threshold_low: Optional[float] = None,
    scars_threshold_high: Optional[float] = None,
    scars_max_scar_width: Optional[int] = None,
    scars_min_scar_length: Optional[int] = None,
) -> MutableMapping[str, Any]:
    """
    Build a TopoStats-compatible filter config from defaults + overrides.

    Explicit keyword args override both defaults and any values provided by
    `filter_config`.
    """
    cfg: MutableMapping[str, Any] = _default_filter_config()

    if filter_config is not None:
        _deep_update(cfg, filter_config)

    # Apply explicit overrides (only when not None)
    if row_alignment_quantile is not None:
        cfg["row_alignment_quantile"] = float(row_alignment_quantile)
    if threshold_method is not None:
        cfg["threshold_method"] = str(threshold_method)
    if otsu_threshold_multiplier is not None:
        cfg["otsu_threshold_multiplier"] = float(otsu_threshold_multiplier)
    if gaussian_size is not None:
        cfg["gaussian_size"] = float(gaussian_size)
    if gaussian_mode is not None:
        cfg["gaussian_mode"] = str(gaussian_mode)

    if threshold_std_dev_above is not None:
        cfg.setdefault("threshold_std_dev", {})["above"] = float(
            threshold_std_dev_above
        )
    if threshold_std_dev_below is not None:
        cfg.setdefault("threshold_std_dev", {})["below"] = float(
            threshold_std_dev_below
        )

    if threshold_absolute_above is not None:
        cfg.setdefault("threshold_absolute", {})["above"] = float(
            threshold_absolute_above
        )
    if threshold_absolute_below is not None:
        cfg.setdefault("threshold_absolute", {})["below"] = float(
            threshold_absolute_below
        )

    # Scar removal nested config
    if remove_scars is not None:
        cfg.setdefault("remove_scars", {})["run"] = bool(remove_scars)
    if scars_removal_iterations is not None:
        cfg.setdefault("remove_scars", {})["removal_iterations"] = int(
            scars_removal_iterations
        )
    if scars_threshold_low is not None:
        cfg.setdefault("remove_scars", {})["threshold_low"] = float(scars_threshold_low)
    if scars_threshold_high is not None:
        cfg.setdefault("remove_scars", {})["threshold_high"] = float(
            scars_threshold_high
        )
    if scars_max_scar_width is not None:
        cfg.setdefault("remove_scars", {})["max_scar_width"] = int(scars_max_scar_width)
    if scars_min_scar_length is not None:
        cfg.setdefault("remove_scars", {})["min_scar_length"] = int(
            scars_min_scar_length
        )

    return cfg


def _build_topostats_class(
    frame: np.ndarray,
    pixel_to_nm_scaling: float,
) -> Optional[Any]:
    """Build a TopoStats object if available; otherwise return None."""
    ts_frame: Optional[Any] = None
    try:
        from topostats.classes import TopoStats

        ts_frame = TopoStats(
            image_original=frame,
            pixel_to_nm_scaling=float(pixel_to_nm_scaling),
            filename="frame",
        )
    except ImportError:
        logger.info("topostats.classes not found; using older TopoStats (< 2.4)")

    return ts_frame


def topostats_filter(
    frame: np.ndarray,
    *,
    # ---- non-filter arguments ----
    pixel_to_nm_scaling: float = 1.0,  # required by TopoStats; unused by this wrapper
    output_key: str = "gaussian_filtered",
    on_failure: str = "return_input",  # "return_input" | "return_none" | "raise"
    # ---- filter configuration file dict ----
    filter_config: Optional[Mapping[str, Any]] = None,
    # ---- explicit CLI/API knobs (forwarded to builder) ----
    row_alignment_quantile: Optional[float] = None,
    threshold_method: Optional[str] = None,
    otsu_threshold_multiplier: Optional[float] = None,
    gaussian_size: Optional[float] = None,
    gaussian_mode: Optional[str] = None,
    threshold_std_dev_above: Optional[float] = None,
    threshold_std_dev_below: Optional[float] = None,
    threshold_absolute_above: Optional[float] = None,
    threshold_absolute_below: Optional[float] = None,
    remove_scars: Optional[bool] = None,
    scars_removal_iterations: Optional[int] = None,
    scars_threshold_low: Optional[float] = None,
    scars_threshold_high: Optional[float] = None,
    scars_max_scar_width: Optional[int] = None,
    scars_min_scar_length: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Apply TopoStats filtering to a single AFM frame, with CLI-friendly options.

    Parameters
    ----------
    frame : ndarray
        2D AFM image frame.
    pixel_to_nm_scaling : float
        Pixel-to-nanometre scaling, passed through to TopoStats.
    output_key : str
        Name of the intermediate/final image in `filters.images` to return.
    on_failure : {"return_input", "return_none", "raise"}
        Behaviour if filtering fails or `output_key` is missing.
    filter_config : Mapping[str, Any] | None
        Optional TopoStats-compatible configuration dict for the filter stage.
        Explicit keyword arguments override values in this dict.
    Other kwargs
        CLI/API-friendly overrides corresponding to TopoStats `filter:` settings.

    Returns
    -------
    ndarray or None
        Filtered frame (or original frame / None depending on `on_failure`).
    """
    # Normalize first and validate before touching TopoStats
    frame = np.asarray(frame)

    if frame.ndim != 2:
        if on_failure == "raise":
            raise ValueError(f"Expected a 2D frame, got shape {frame.shape}")
        return None if on_failure == "return_none" else frame

    if frame.size == 0:
        if on_failure == "raise":
            raise ValueError("Empty frame provided")
        return None if on_failure == "return_none" else frame

    if np.isnan(frame).any():
        if on_failure == "raise":
            raise ValueError("NaNs present in frame")
        return None if on_failure == "return_none" else frame

    # Try to import Filters (required)
    try:
        from topostats.filters import Filters
    except ImportError as e:
        # Keep the message actionable
        raise ImportError(
            "TopoStats is required for topostats_filter. "
            "Install with: pip install playnano-plugins[topostats]"
        ) from e

    ts_frame = _build_topostats_class(frame, pixel_to_nm_scaling)

    # Build configuration
    config = _build_filter_config(
        filter_config=filter_config,
        row_alignment_quantile=row_alignment_quantile,
        threshold_method=threshold_method,
        otsu_threshold_multiplier=otsu_threshold_multiplier,
        gaussian_size=gaussian_size,
        gaussian_mode=gaussian_mode,
        threshold_std_dev_above=threshold_std_dev_above,
        threshold_std_dev_below=threshold_std_dev_below,
        threshold_absolute_above=threshold_absolute_above,
        threshold_absolute_below=threshold_absolute_below,
        remove_scars=remove_scars,
        scars_removal_iterations=scars_removal_iterations,
        scars_threshold_low=scars_threshold_low,
        scars_threshold_high=scars_threshold_high,
        scars_max_scar_width=scars_max_scar_width,
        scars_min_scar_length=scars_min_scar_length,
    )

    try:
        if ts_frame is not None:
            filters = Filters(topostats_object=ts_frame, **config)
        else:
            filters = Filters(
                image=frame,
                pixel_to_nm_scaling=float(pixel_to_nm_scaling),
                filename="frame",
                **config,
            )

        filters.filter_image()

        out = filters.images.get(output_key)
        if out is None:
            msg = f"TopoStats output '{output_key}' not found in filters.images."
            if on_failure == "raise":
                raise KeyError(msg)
            logger.warning(msg)
            return None if on_failure == "return_none" else frame

        return np.asarray(out, dtype=float)

    except Exception as e:
        if on_failure == "raise":
            raise
        logger.warning(f"Exception in topostats_filter: {e}")
        return None if on_failure == "return_none" else frame


topostats_filter.__version__ = "0.1.1"  # type: ignore[attr-defined]
