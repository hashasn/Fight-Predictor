import numpy as np
import pandas as pd
import pytest
import torch
import train
from train import chronological_split_3way, FightNet, train_nn


# chronological_split_3way

def make_paired(n_fights):
    """
    build a small paired dataframe for testing.
    each fight has 2 rows (one from each fighter's perspective),
    like the real paired dataset
    """
    rows = []
    for fight_id in range(n_fights):
        rows.append({"fight_id": fight_id, "elo_diff_pre": 10.0, "result_f": 1})
        rows.append({"fight_id": fight_id, "elo_diff_pre": -10.0, "result_f": 0})

    # shuffle the rows so the test doesn't rely on them already being in order
    return pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)


def test_chronological_split_3way():
    """
    oldest fights -> train, middle -> val, newest -> test.
    no fight should ever appear in more than one split
    """
    paired = make_paired(100)
    feature_cols = ["elo_diff_pre"]

    (train_df, val_df, test_df,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test) = chronological_split_3way(
        paired=paired,
        feature_cols=feature_cols,
        test_size=0.2,
        val_size=0.2,
    )

    train_ids = set(train_df["fight_id"])
    val_ids = set(val_df["fight_id"])
    test_ids = set(test_df["fight_id"])

    # 100 fights: newest 20 go to test,
    # then 20% of the remaining 80 (16) go to val, leaving 64 for train
    assert len(train_ids) == 64
    assert len(val_ids) == 16
    assert len(test_ids) == 20

    # no fight is in more than one split
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # every fight ends up somewhere
    assert train_ids | val_ids | test_ids == set(range(100))

    # chronological: all train fights are older than all val fights,
    # and all val fights are older than all test fights
    assert max(train_ids) < min(val_ids)
    assert max(val_ids) < min(test_ids)

    # both rows of a fight stay together (2 rows per fight)
    assert len(train_df) == 64 * 2
    assert len(val_df) == 16 * 2
    assert len(test_df) == 20 * 2

    # X only has the feature columns, y is the result
    assert list(X_train.columns) == feature_cols
    assert y_train.tolist() == train_df["result_f"].tolist()
    assert len(X_test) == len(y_test)



    # FightNet

def test_fightnet_output_shape():
    """
    FightNet takes a batch of feature rows and returns
    one logit (raw score) per row
    """
    torch.manual_seed(42)
    model = FightNet(16)
    model.eval()   # turn off dropout so results are repeatable

    # batch of 4 fights, 16 features each -> 4 outputs
    x = torch.randn(4, 16)
    out = model(x)
    assert out.shape == (4,)

    # a single fight still works (this is what predict.py does)
    single = model(torch.randn(1, 16))
    assert single.shape == (1,)

    # sigmoid turns the logits into probabilities between 0 and 1
    probs = torch.sigmoid(out)
    assert ((probs > 0) & (probs < 1)).all()

    # in eval mode the same input always gives the same output
    assert torch.equal(model(x), model(x))



    # train_nn

def test_train_nn_keeps_best_epoch(monkeypatch):
    """
    early stopping should restore the weights from the best epoch,
    so the final val auc should equal the best val auc seen in training
    """
    # record every auc that train_nn calculates
    auc_calls = []
    real_roc_auc_score = train.roc_auc_score

    def recording_roc_auc_score(y_true, y_score):
        auc = real_roc_auc_score(y_true, y_score)
        auc_calls.append(auc)
        return auc

    monkeypatch.setattr(train, "roc_auc_score", recording_roc_auc_score)

    # random data with no real pattern, so val auc jumps around
    # between epochs and early stopping kicks in
    rng = np.random.default_rng(0)
    n_features = 16
    X_train = rng.normal(size=(300, n_features)).astype(np.float32)
    y_train = rng.integers(0, 2, size=300).astype(np.float32)
    X_val = rng.normal(size=(100, n_features)).astype(np.float32)
    y_val = rng.integers(0, 2, size=100).astype(np.float32)
    X_test = rng.normal(size=(100, n_features)).astype(np.float32)
    y_test = rng.integers(0, 2, size=100).astype(np.float32)

    model, history = train_nn(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_cols=[f"f{i}" for i in range(n_features)],
    )

    # the last 2 auc calls are the final val and test evaluations,
    # everything before that is one call per epoch
    epoch_aucs = auc_calls[:-2]
    best_epoch_auc = max(epoch_aucs)

    assert history["val_auc"] == pytest.approx(best_epoch_auc)