import pandas as pd

from audiovocab.splits import check_grouped, folds, hash_fold


def _df():
    return pd.DataFrame({
        "dataset": ["esc50", "esc50", "fsd50k", "fsd50k", "fsd50k", "xenocanto"],
        "split": ["1", "3", "eval", "eval", "eval", ""],
        "group_id": ["src_a", "src_b", "up_x", "up_x", "up_y", "rec_1"],
    })


def test_esc50_keeps_official_folds():
    f = folds(_df())
    assert f.iloc[0] == 1 and f.iloc[1] == 3


def test_same_group_same_fold_and_stable():
    df = _df()
    f = folds(df)
    assert f.iloc[2] == f.iloc[3] == hash_fold("up_x")
    assert check_grouped(df, f)
    assert (folds(df.iloc[::-1]).sort_index() == f).all()  # independent of row order


def test_folds_in_range():
    f = folds(pd.DataFrame({"dataset": ["x"] * 200, "split": [""] * 200, "group_id": [f"g{i}" for i in range(200)]}))
    assert set(f) == {1, 2, 3, 4, 5}


def test_check_grouped_detects_leak():
    df = _df()
    bad = [1, 2, 3, 4, 5, 1]  # up_x split across folds 3 and 4
    assert not check_grouped(df, bad)
