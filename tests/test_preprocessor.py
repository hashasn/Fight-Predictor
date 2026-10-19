import numpy as np
import pandas as pd
import pytest
from preprocessor import mmss_to_seconds, height_to_inches, reach_to_inches, parse_of, get_result, method_group, elo_expected, build_elo_features, add_rolling_features, derandomise_fighter_order
# mmss_to_seconds tests

def test_mmss_to_seconds():
    #convert a 'm:ss' string into total seconds
    # invalid or missing values should return NaN

    #valid times
    assert mmss_to_seconds("4:32") == 272
    assert mmss_to_seconds("0:00") == 0
    assert mmss_to_seconds("5:00") == 300
    assert mmss_to_seconds(" 1:05 ") == 65

    #invalid times
    assert np.isnan(mmss_to_seconds("abc"))
    assert np.isnan(mmss_to_seconds("4:5"))    # seconds must be 2 digits
    assert np.isnan(mmss_to_seconds(""))
    assert np.isnan(mmss_to_seconds(None))
    assert np.isnan(mmss_to_seconds(np.nan))

# height_to_inches

def test_height_to_inches():
    """
    convert a height string like 5' 11" into inches.
    invalid or missing values should return nan
    """
    # valid heights
    assert height_to_inches("5' 11\"") == 71
    assert height_to_inches("6' 0\"") == 72
    assert height_to_inches("5'9\"") == 69      # no space after the '

    # invalid heights
    assert np.isnan(height_to_inches("--"))
    assert np.isnan(height_to_inches("6'"))     # missing inches
    assert np.isnan(height_to_inches(None))
    assert np.isnan(height_to_inches(np.nan))


# reach_to_inches

def test_reach_to_inches():
    """
    convert a reach string like 72" into numeric inches.
    invalid or missing values should return nan
    """
    # valid reaches
    assert reach_to_inches('72"') == 72.0
    assert reach_to_inches('74.5"') == 74.5
    assert reach_to_inches("70") == 70.0        # no " symbol

    # invalid reaches
    assert np.isnan(reach_to_inches("--"))
    assert np.isnan(reach_to_inches(""))
    assert np.isnan(reach_to_inches(None))


# parse_of

def test_parse_of():
    """
    convert a stat string like '45 of 102' into (landed, attempted, rate).
    invalid or missing values should return (nan, nan, nan)
    """
    # valid stats
    landed, attempted, rate = parse_of("45 of 102")
    assert landed == 45
    assert attempted == 102
    assert rate == pytest.approx(45 / 102)

    landed, attempted, rate = parse_of("10 of 10")
    assert rate == 1.0

    # 0 attempts should give a rate of 0, not a divide by zero error
    landed, attempted, rate = parse_of("0 of 0")
    assert landed == 0
    assert attempted == 0
    assert rate == 0.0

    # invalid stats
    for value in ["--", "abc", "x of y", "", None, np.nan]:
        landed, attempted, rate = parse_of(value)
        assert np.isnan(landed)
        assert np.isnan(attempted)
        assert np.isnan(rate)


# get_result

def test_get_result():
    """
    result from one fighter's perspective.
    1 -> won, 0 -> lost, nan -> draw / no contest / unknown
    """
    # win and loss
    assert get_result("Jean Silva", "Jean Silva") == 1
    assert get_result("Jean Silva", "Jose Delgado") == 0
    assert get_result(" Jean Silva ", "Jean Silva") == 1     # whitespace is stripped

    # draw, no contest or missing winner
    for winner in ["DRAW", "draw", "NO CONTEST", "NC", "", None, np.nan]:
        assert np.isnan(get_result(winner, "Jean Silva"))


# method_group

def test_method_group():
    """
    collapse detailed method labels into ko / sub / dec / other
    """
    assert method_group("KO/TKO") == "ko"
    assert method_group("Submission") == "sub"
    assert method_group("Decision - Unanimous") == "dec"
    assert method_group("Decision - Split") == "dec"
    assert method_group("Decision - Majority") == "dec"

    # anything else falls into other
    assert method_group("DQ") == "other"
    assert method_group("Overturned") == "other"
    assert method_group("") == "other"
    assert method_group(None) == "other"
    assert method_group("TKO - Doctor's Stoppage") == "other"



# elo_expected

def test_elo_expected():
    """
    standard elo expected score formula.
    returns the chance fighter a beats fighter b
    """
    # equal ratings is a coin flip
    assert elo_expected(1500, 1500) == 0.5

    # 400 points higher -> 10 to 1 favourite
    assert elo_expected(1900, 1500) == pytest.approx(10 / 11)

    # higher rated fighter is the favourite
    assert elo_expected(1600, 1500) > 0.5
    assert elo_expected(1500, 1600) < 0.5

    # both sides should add up to 1
    assert elo_expected(1650, 1520) + elo_expected(1520, 1650) == pytest.approx(1.0)


# build_elo_features

def make_fights(rows):
    """
    build a small fight level dataframe for testing.
    each row is (fighter1, fighter2, winner)
    """
    return pd.DataFrame({
        "fight_index": range(len(rows)),
        "Fighter1": [r[0] for r in rows],
        "Fighter2": [r[1] for r in rows],
        "Winner": [r[2] for r in rows]
    })


def test_build_elo_features():
    """
    elo ratings should update after wins and draws,
    and stay the same after no contest
    """

    df = make_fights([
        ("A", "B", "A"),   # A beats B
        ("A", "C", "DRAW"),# draw between B and C
        ("A", "B", "NO CONTEST")   # no contest between A and B
    ])

    d = build_elo_features(df, base_elo=1500, k=16)

    # fight 0: both new fighters start at 1500, so it's a 50/50 fight.
    # winner gains k * 0.5 = 8, loser drops 8
    win = d.iloc[0]
    assert win["elo_pre_f1"] == 1500
    assert win["elo_pre_f2"] == 1500
    assert win["elo_diff_pre"] == 0
    assert win["elo_post_f1"] == 1508
    assert win["elo_post_f2"] == 1492
    assert win["s1"] == 1.0

    # fight 1: A carries their new rating into the next fight
    draw = d.iloc[1]
    assert draw["elo_pre_f1"] == 1508
    assert draw["elo_pre_f2"] == 1500
    assert draw["s1"] == 0.5

    # A was the favourite, so a draw should cost A points and give C points
    expected_a = elo_expected(1508, 1500)
    assert draw["elo_post_f1"] == pytest.approx(1508 + 16 * (0.5 - expected_a))
    assert draw["elo_post_f1"] < 1508
    assert draw["elo_post_f2"] > 1500

    # fight 2: no contest, ratings don't change
    nc = d.iloc[2]
    assert nc["elo_post_f1"] == nc["elo_pre_f1"]
    assert nc["elo_post_f2"] == nc["elo_pre_f2"]
    assert np.isnan(nc["s1"])

    # elo is zero sum: points gained by one fighter are lost by the other
    for _, row in d.iterrows():
        total_pre = row["elo_pre_f1"] + row["elo_pre_f2"]
        total_post = row["elo_post_f1"] + row["elo_post_f2"]
        assert total_post == pytest.approx(total_pre)



# add_rolling_features

def make_history(fighter, fight_indexes, results, sig_rates):
    """
    build a samll fighter-history dataframe for testing.
    only result and sig_rate change, every other stat is a constant
    """
    n = len(fight_indexes)

    return pd.DataFrame({
        "fighter": [fighter]*n,
        "fight_index": fight_indexes,
        "weightclass": ["Lightweight"] * n,
        "result": results,
        "sig_rate": sig_rates,
        "sig_per_sec": [0.1] * n,
        "ctrl_pct": [0.2] * n,
        "td_rate": [0.3] * n,
        "kd_per_sec": [0.0] * n,
        "opp_elo_pre": [1500] * n,
        "win_ko": [0] * n,
        "win_sub": [0] * n,
    })


def test_add_rolling_features_no_leakage():
    """
    rolling features should only use fights before the current one.
    a fighters own stats should not be used to predict their next fight.
    """

    #fighter A has 7 fights,fighter B has 2 fights in betrween
    a_hist = make_history(
        "A", 
        fight_indexes=[0,2,4,6,8,10,12],
        results=[1,1,0,1,1,1,0],
        sig_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )

    b_hist = make_history(
        "B", 
        fight_indexes=[1, 3],
        results=[0, 1],
        sig_rates=[0.9, 0.9],
    )

    hist = pd.concat([a_hist, b_hist], ignore_index=True)
    print(hist.sort_values("fight_index"))
    d = add_rolling_features(hist, window=5)
    a = d[d["fighter"] == "A"].reset_index(drop=True)
    b = d[d["fighter"] == "B"].reset_index(drop=True)

    # first 5 fights don't have 5 previous fights yet, so no average
    assert a["sig_rate_avg"].iloc[:5].isna().all()

    # fight 6 uses fights 1-5 only: mean(0.1, 0.2, 0.3, 0.4, 0.5) = 0.3
    # if the current fight leaked in it would be mean(0.2 ... 0.6) = 0.4
    assert a["sig_rate_avg"].iloc[5] == pytest.approx(0.3)

    # fight 7 uses fights 2-6 only, not its own 0.7
    assert a["sig_rate_avg"].iloc[6] == pytest.approx(0.4)

    # experience counts previous fights, so a debut is 0
    assert a["exp_prior"].tolist() == [0, 1, 2, 3, 4, 5, 6]

    # streaks going INTO each fight, not including that fight's result
    # results:  W  W  L  W  W  W  L
    assert np.isnan(a["win_streak_prior"].iloc[0])
    assert a["win_streak_prior"].iloc[1:].tolist() == [1, 2, 0, 1, 2, 3]
    assert a["loss_streak_prior"].iloc[1:].tolist() == [0, 0, 1, 0, 0, 0]

    # B's fights are tracked separately and never mix with A's
    assert b["exp_prior"].tolist() == [0, 1]
    assert b["sig_rate_avg"].isna().all()


# derandomise_fighter_order

def make_corner_fights(n):
    """
    build n fights where the red corner (Fighter1) always wins.
    red stats are always 1 / 72", blue stats are always 0 / 70"
    """
    return pd.DataFrame({
        "Event": ["UFC Test"] * n,
        "Fighter1": [f"red{i}" for i in range(n)],
        "Fighter2": [f"blue{i}" for i in range(n)],
        "Winner": [f"red{i}" for i in range(n)],
        "KD1": [1] * n,
        "KD2": [0] * n,
        "f1_reach": ['72"'] * n,
        "f2_reach": ['70"'] * n,
    })


def test_derandomise_fighter_order():
    """
    about half the fights should have their corners swapped,
    and every f1/f2 column must swap together with the names
    """
    df = make_corner_fights(200)
    original = df.copy()

    swapped_df = derandomise_fighter_order(df, seed=42)

    # the original dataframe is not modified
    assert df.equals(original)

    # same seed gives the same result every time
    assert swapped_df.equals(derandomise_fighter_order(df, seed=42))

    # red no longer always wins: Fighter1 should win roughly half the time
    f1_win_share = (swapped_df["Winner"] == swapped_df["Fighter1"]).mean()
    assert 0.4 < f1_win_share < 0.6

    for i, row in swapped_df.iterrows():
        # Winner is a name, so it should never change
        assert row["Winner"] == f"red{i}"

        if row["Fighter1"] == f"red{i}":
            # not swapped: everything stays in place
            assert row["Fighter2"] == f"blue{i}"
            assert row["KD1"] == 1 and row["KD2"] == 0
            assert row["f1_reach"] == '72"' and row["f2_reach"] == '70"'
        else:
            # swapped: every red stat moves with the red fighter
            assert row["Fighter1"] == f"blue{i}"
            assert row["Fighter2"] == f"red{i}"
            assert row["KD1"] == 0 and row["KD2"] == 1
            assert row["f1_reach"] == '70"' and row["f2_reach"] == '72"'

    # columns without a 1/2 pair are left alone
    assert (swapped_df["Event"] == "UFC Test").all()

