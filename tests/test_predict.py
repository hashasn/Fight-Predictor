import numpy as np
import pandas as pd
import pytest

from predict import get_current_form


# get_current_form

def make_rolled_history(fighter, results, sig_rates, win_streaks, loss_streaks):
    """
    build a small fighter_hist_rolled dataframe for testing.
    only the columns get_current_form reads are included
    """
    n = len(results)
    return pd.DataFrame({
        "fighter": [fighter] * n,
        "fight_index": range(n),
        "result": results,
        "sig_rate": sig_rates,
        "sig_per_sec": [0.1] * n,
        "ctrl_pct": [0.2] * n,
        "td_rate": [0.3] * n,
        "kd_per_sec": [0.0] * n,
        "win_ko": [0] * n,
        "win_sub": [0] * n,
        "exp_prior": range(n),
        "exp_div_prior": range(n),
        "win_streak": win_streaks,
        "loss_streak": loss_streaks,
        "own_elo_pre": [1500 + 10 * i for i in range(n)],
    })


def test_get_current_form():
    """
    form going INTO a new fight should use the fighter's most recent
    fights, including their last one (it has already happened)
    """
    # A: 6 fights, won the last 3
    a_hist = make_rolled_history(
        "A",
        results=[1, 1, 0, 1, 1, 1],
        sig_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        win_streaks=[1, 2, 0, 1, 2, 3],
        loss_streaks=[0, 0, 1, 0, 0, 0],
    )
    # B: 2 fights, lost the last one
    b_hist = make_rolled_history(
        "B",
        results=[1, 0],
        sig_rates=[0.5, 0.7],
        win_streaks=[1, 0],
        loss_streaks=[0, 1],
    )
    hist = pd.concat([a_hist, b_hist], ignore_index=True)

    form_a = get_current_form("A", hist, window=5)

    # last 5 fights INCLUDING the most recent: mean(0.2, 0.3, 0.4, 0.5, 0.6)
    assert form_a["sig_rate_avg"] == pytest.approx(0.4)

    # last 5 results: W L W W W -> 4 out of 5
    assert form_a["win_rate_avg"] == pytest.approx(0.8)

    # 5 fights before the last one + the last one = 6 fights going in
    assert form_a["exp_prior"] == 6

    # won the last fight, so the 3 fight win streak carries into the next one
    assert form_a["win_streak_prior"] == 3
    assert form_a["loss_streak_prior"] == 0

    # B lost their last fight, so it's a loss streak going in
    form_b = get_current_form("B", hist, window=5)
    assert form_b["win_streak_prior"] == 0
    assert form_b["loss_streak_prior"] == 1

    # B only has 2 fights, so the average uses just those 2
    assert form_b["sig_rate_avg"] == pytest.approx(0.6)

    # a fighter with no history should raise an error
    with pytest.raises(ValueError, match="No history found"):
        get_current_form("Nobody", hist, window=5)