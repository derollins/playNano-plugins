import numpy as np
import pytest

from playnano_plugins.processing.topostats_filter import (
    _build_filter_config,
    _deep_update,
    _default_filter_config,
    topostats_filter,
)


def test_deep_update_merges_nested():
    dst = {"a": {"x": 1, "y": 2}, "b": 3}
    src = {"a": {"y": 99}, "c": 4}
    out = _deep_update(dst, src)
    assert out["a"]["x"] == 1
    assert out["a"]["y"] == 99
    assert out["b"] == 3
    assert out["c"] == 4


def test_build_filter_config_defaults_and_overrides():
    cfg = _build_filter_config()
    dflt = _default_filter_config()
    assert cfg["threshold_method"] == dflt["threshold_method"]
    assert cfg["remove_scars"]["run"] == dflt["remove_scars"]["run"]

    # override via filter_config dict
    cfg2 = _build_filter_config(filter_config={"gaussian_size": 9.0})
    assert cfg2["gaussian_size"] == 9.0

    # explicit kwargs override dict
    cfg3 = _build_filter_config(
        filter_config={"gaussian_size": 9.0, "remove_scars": {"run": False}},
        gaussian_size=2.0,
        remove_scars=True,
        scars_threshold_low=0.123,
    )
    assert cfg3["gaussian_size"] == 2.0
    assert cfg3["remove_scars"]["run"] is True
    assert cfg3["remove_scars"]["threshold_low"] == pytest.approx(0.123)


def test_topostats_filter_rejects_non_2d():
    frame = np.zeros((10, 10, 2))
    with pytest.raises(ValueError, match="Expected a 2D frame"):
        topostats_filter(frame)


def test_topostats_filter_nan_handling_return_input():
    frame = np.zeros((10, 10))
    frame[0, 0] = np.nan
    out = topostats_filter(frame, on_failure="return_input")
    # should return original frame unchanged when NaNs are present
    assert out is frame or np.array_equal(out, frame)


def test_topostats_filter_nan_handling_return_none():
    frame = np.zeros((10, 10))
    frame[0, 0] = np.nan
    out = topostats_filter(frame, on_failure="return_none")
    assert out is None


def test_topostats_filter_nan_handling_raise():
    frame = np.zeros((10, 10))
    frame[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaNs present"):
        topostats_filter(frame, on_failure="raise")


def test_topostats_filter_importerror_message_if_topostats_missing():
    # If TopoStats is installed this test isn't valid; skip in that case.
    try:
        import topostats  # noqa: F401

        pytest.skip("TopoStats installed; ImportError path not applicable.")
    except Exception:
        pass

    frame = np.zeros((10, 10))
    with pytest.raises(ImportError, match=r"pip install playnano-plugins\[topostats\]"):
        topostats_filter(frame)


def test_build_filter_config_scar_overrides():
    cfg = _build_filter_config(
        remove_scars=True,
        scars_removal_iterations=3,
        scars_threshold_low=0.2,
        scars_threshold_high=0.7,
    )
    assert cfg["remove_scars"]["run"] is True
    assert cfg["remove_scars"]["removal_iterations"] == 3
    assert cfg["remove_scars"]["threshold_low"] == 0.2
    assert cfg["remove_scars"]["threshold_high"] == 0.7
