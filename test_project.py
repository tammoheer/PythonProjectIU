# test_project.py
import math
import pandas as pd

from project import FunctionSelector, TestMapper


def make_small_data():
    """Create small artificial datasets (same x-values) so that tests are deterministic."""
    x = pd.Series([0, 1, 2, 3], name="x")
    train = pd.DataFrame({
        "x": x,
        "y1": pd.Series([0, 1, 2, 3]),
        "y2": pd.Series([0, -1, -2, -3]),
        "y3": pd.Series([1, 1, 1, 1]),
        "y4": pd.Series([0, 0, 0, 0]),
    })
    ideal = pd.DataFrame({
        "x": x,
        "y1": pd.Series([0, 1, 2, 3]),     # perfect for train.y1
        "y2": pd.Series([0, -1, -2, -3]),  # perfect for train.y2
        "y3": pd.Series([1, 1, 1, 1]),     # perfect for train.y3
        "y4": pd.Series([0, 0, 0, 0]),     # perfect for train.y4
        "y5": pd.Series([10, 10, 10, 10])  # outlier function
    })
    test = pd.DataFrame({
        "x": x,
        "y": pd.Series([0, 1, 2, 3])  # lies exactly on ideal.y1
    })
    return train, ideal, test


def test_select_best_perfect_match():
    """Verify that the FunctionSelector correctly finds perfect 1:1 mappings."""
    train, ideal, _ = make_small_data()
    sel = FunctionSelector(train, ideal)
    mapping, maxdev = sel.select_best()

    # Expected: y1->y1, y2->y2, y3->y3, y4->y4
    assert mapping["y1"] == "y1"
    assert mapping["y2"] == "y2"
    assert mapping["y3"] == "y3"
    assert mapping["y4"] == "y4"

    # Perfect matches => maxdev == 0
    for ty in ["y1", "y2", "y3", "y4"]:
        assert maxdev[ty] == 0.0


def test_mapping_threshold_rule():
    """Check that the threshold rule correctly includes matching test points."""
    train, ideal, test = make_small_data()
    sel = FunctionSelector(train, ideal)
    mapping, maxdev = sel.select_best()

    mapper = TestMapper(train, ideal, test)
    mapped = mapper.map_points(mapping, maxdev)

    # All test points lie exactly on ideal.y1 => all should be mapped
    assert len(mapped) == len(test)

    # Deviation = 0, threshold = sqrt(2)*0 = 0 -> still allowed (<= 0)
    assert math.isclose(mapped["delta_y"].max(), 0.0)
    assert set(mapped["ideal_func"]) == {"y1"}
