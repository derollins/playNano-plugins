import numpy as np
import pytest

from playnano_plugins.analysis.particle_boundary_size import BoundarySizeModule


class DummyStack:
    """Dummy image stack with time_for_frame method."""

    def time_for_frame(self, i: int) -> float:
        """Dummy time_for_frame method: returns 0.1 * frame index."""
        return 0.1 * float(i)


def _make_previous_results_happy():
    # 3 frames
    labeled_masks = []
    features_per_frame = []
    for f in range(3):
        lm = np.zeros((20, 20), dtype=int)
        # label 1: bbox height grows with frame
        lm[2 : 2 + (3 + f), 5:9] = 1  # width ~4, height ~3+f
        labeled_masks.append(lm)

        # one feature per frame at index 0, with label 1
        features_per_frame.append([{"label": 1}])

    # one track with frames aligned and pt_idx=0 always
    tracks = [{"id": 0, "frames": [0, 1, 2], "point_indices": [0, 0, 0]}]

    return {
        "feature_detection": {
            "labeled_masks": labeled_masks,
            "features_per_frame": features_per_frame,
        },
        "particle_tracking": {"tracks": tracks},
    }


def test_boundary_size_happy_path_with_threshold():
    prev = _make_previous_results_happy()
    m = BoundarySizeModule()
    out = m.run(DummyStack(), prev, threshold=4)  # threshold between some frames

    assert out["measure"] == "bbox_max_dim"
    assert out["threshold"] == 4
    assert "flat_table" in out and len(out["flat_table"]) == 3
    assert "per_track" in out and len(out["per_track"]) == 1

    # Check required columns exist
    row0 = out["flat_table"][0]
    assert {"track_id", "label", "frame", "timestamp", "max_dim", "state"}.issubset(
        row0
    )

    # label should be present and numeric
    assert row0["label"] == 1

    # state should be 0/1 or nan
    states = [r["state"] for r in out["flat_table"]]
    assert all((s in (0, 1) or (isinstance(s, float) and np.isnan(s))) for s in states)

    # Summary bookkeeping
    assert out["summary"]["n_tracks"] == 1
    assert out["summary"]["n_rows"] == 3
    assert out["summary"]["state_included"] is True


def test_boundary_size_out_of_range_point_index_writes_nans():
    prev = _make_previous_results_happy()
    # make pt_idx invalid on frame 1
    prev["particle_tracking"]["tracks"][0]["point_indices"] = [0, 99, 0]

    m = BoundarySizeModule()
    with pytest.warns(UserWarning, match="point_index 99 out of range"):
        out = m.run(DummyStack(), prev, threshold=None)

    rows = out["flat_table"]
    assert len(rows) == 3

    r1 = rows[1]
    assert r1["frame"] == 1
    assert np.isnan(r1["label"])
    assert np.isnan(r1["max_dim"])

    assert out["summary"]["n_skipped_index_errors"] >= 1


def test_boundary_size_missing_label_raises():
    prev = _make_previous_results_happy()
    # remove label key from the feature dict in frame 0
    prev["feature_detection"]["features_per_frame"][0][0] = {}

    m = BoundarySizeModule()
    with pytest.raises(RuntimeError, match="does not contain 'label'"):
        m.run(DummyStack(), prev)


def test_boundary_size_measure_mismatch_raises():
    m = BoundarySizeModule()
    with pytest.raises(ValueError, match="only supports measure"):
        m.run(
            DummyStack(),
            {
                "particle_tracking": {"tracks": []},
                "feature_detection": {"labeled_masks": [], "features_per_frame": []},
            },
            measure="other",
        )


def test_boundary_size_missing_tracking_module_raises():
    m = BoundarySizeModule()
    with pytest.raises(RuntimeError, match="requires tracking_module"):
        m.run(
            DummyStack(),
            {"feature_detection": {"labeled_masks": [], "features_per_frame": []}},
        )


def test_boundary_size_frame_out_of_range_warns_and_nans():
    prev = {
        "feature_detection": {
            "labeled_masks": [np.zeros((10, 10), dtype=int)],
            "features_per_frame": [[{"label": 1}]],
        },
        "particle_tracking": {
            "tracks": [{"id": 0, "frames": [99], "point_indices": [0]}],
        },
    }
    m = BoundarySizeModule()
    with pytest.warns(UserWarning, match="frame 99 out of range"):
        out = m.run(DummyStack(), prev)
    row = out["flat_table"][0]
    assert np.isnan(row["label"])
    assert np.isnan(row["max_dim"])
