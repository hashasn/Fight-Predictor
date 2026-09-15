import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessor import run_preprocessor, BASE_FEATURE_COLS, FINAL_FEATURE_COLS
from train import FightNet, BASE_ELO, ELO_K, ROLLING_WINDOW

ARTIFACTS_DIR = Path("artifacts")
feature_cols = None
preprocessor = None
model = None

# load feature_cols / preprocessor / model from artifacts inti this
# moduls globals
def reload_artifacts():
    global feature_cols, preprocessor, model
    with open(ARTIFACTS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")

    model = FightNet(len(feature_cols))
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "fightnet_state.pt"))
    model.eval()




def get_current_form(fighter_name, fighter_hist_rolled, window=ROLLING_WINDOW):
    """
    Recompute this fighter's rolling stats fresh, using their most recent
    `window` completed fights -- i.e. their form heading INTO a new fight,
    not their form heading into their last recorded one.
    """
    rows = fighter_hist_rolled[fighter_hist_rolled["fighter"] == fighter_name]
    if rows.empty:
        raise ValueError(f"No history found for fighter: {fighter_name}")

    rows = rows.sort_values("fight_index")
    recent = rows.tail(window)

    raw_cols = {
        "sig_per_sec_avg": "sig_per_sec",
        "sig_rate_avg": "sig_rate",
        "ctrl_pct_avg": "ctrl_pct",
        "td_rate_avg": "td_rate",
        "win_rate_avg": "result",
        "kd_per_sec_avg": "kd_per_sec",
        "ko_win_rate_avg": "win_ko",
        "sub_win_rate_avg": "win_sub",
    }

    form = {}
    for avg_col, raw_col in raw_cols.items():
        form[avg_col] = recent[raw_col].mean()

    latest = rows.iloc[-1]
    form["exp_prior"] = latest["exp_prior"] + 1          # +1 since they've had one more fight since
    form["exp_div_prior"] = latest["exp_div_prior"] + 1   # approximation -- see note below
    form["win_streak_prior"] = latest["win_streak"] if latest["result"] == 1 else 0
    form["loss_streak_prior"] = latest["loss_streak"] if latest["result"] == 0 else 0

    # static / latest-known bio info
    form["own_elo_current"] = latest["own_elo_pre"]  # see elo note below
    form["fight_index_latest"] = latest["fight_index"]

    return form


def get_bio(fighter_name, df_fight):
    """
    Look up a fighter's most recent known age/height/reach/elo from df_fight.
    """
    as_f1 = df_fight[df_fight["Fighter1"] == fighter_name]
    as_f2 = df_fight[df_fight["Fighter2"] == fighter_name]

    if as_f1.empty and as_f2.empty:
        raise ValueError(f"No fight-level record found for: {fighter_name}")

    records = []
    for _, r in as_f1.iterrows():
        records.append({
            "fight_index": r["fight_index"],
            "dob": r["f1_dob"],
            "height_in": r["f1_height_in"],
            "reach_in": r["f1_reach_in"],
            "elo_post": r["elo_post_f1"],
        })
    for _, r in as_f2.iterrows():
        records.append({
            "fight_index": r["fight_index"],
            "dob": r["f2_dob"],
            "height_in": r["f2_height_in"],
            "reach_in": r["f2_reach_in"],
            "elo_post": r["elo_post_f2"],
        })

    bio_df = pd.DataFrame(records).sort_values("fight_index")
    return bio_df.iloc[-1]  # most recent known bio + elo


def predict_matchup(fighter_a, fighter_b, bundle, fight_date=None):
    fighter_hist_rolled = bundle["fighter_hist_rolled"]
    df_fight = bundle["df_fight"]

    if fight_date is None:
        fight_date = pd.Timestamp.now()

    form_a = get_current_form(fighter_a, fighter_hist_rolled)
    form_b = get_current_form(fighter_b, fighter_hist_rolled)

    bio_a = get_bio(fighter_a, df_fight)
    bio_b = get_bio(fighter_b, df_fight)

    age_a = (fight_date - bio_a["dob"]).days / 365.25
    age_b = (fight_date - bio_b["dob"]).days / 365.25

    features = {}
    for col in BASE_FEATURE_COLS:
        features[f"{col}_diff"] = form_a[col] - form_b[col]

    features["age_diff"] = age_a - age_b
    features["height_diff"] = bio_a["height_in"] - bio_b["height_in"]
    features["reach_diff"] = bio_a["reach_in"] - bio_b["reach_in"]
    features["elo_diff_pre"] = bio_a["elo_post"] - bio_b["elo_post"]

    X = pd.DataFrame([features])[feature_cols]

    X_scaled = preprocessor.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logit = model(X_t)
        prob = torch.sigmoid(logit).item()

    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "fighter_a_win_prob": round(prob, 3),
        "fighter_b_win_prob": round(1 - prob, 3),
    }

# load one at import
reload_artifacts()
