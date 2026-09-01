import numpy as np
import pandas as pd
import re

# helpers



def mmss_to_seconds(value):
    """
    Convert a time string like '4:32' into total seconds.
    Returns np.nan if the value is missing or invalid.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    match = re.match(r"^(\d+):(\d{2})$", value)
    if not match:
        return np.nan

    minutes = int(match.group(1))
    seconds = int(match.group(2))
    # print(minutes * 60 + seconds)
    return minutes * 60 + seconds


def height_to_inches(value):
    """
    convert a height string like 5' 11 into inches.
    returns np.nan if missing or invalid
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    match = re.search(r"(\d+)'\s*(\d+)", value)
    if not match:
        return np.nan

    feet = int(match.group(1))
    inches = int(match.group(2))
    return feet * 12 + inches

def reach_to_inches(value):
    """
    convert reach string like 72" into numeric inches
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace('"', '')
    if value in ["--", "", "nan", "None"]:
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", value)
    return float(match.group()) if match else np.nan

def clean_stance(value):
    """
    Normalize stance values into a small set of categories
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()

    if "southpaw" in s:
        return "Southpaw"
    if "orthodox" in s:
        return "Orthodox"
    if "switch" in s:
        return "Switch"

    return "Unknown"

def parse_of(value):
    """
    Convert a stat string like '45 of 102' into:
    landed, attempted, rate

    Returns:
        (landed, attempted, rate)
    or:
        (np.nan, np.nan, np.nan)
    """
    if pd.isna(value):
        return np.nan, np.nan, np.nan

    value = str(value).strip()
    if "of" not in value:
        return np.nan, np.nan, np.nan

    try:
        landed, attempted = value.split("of")
        landed = int(landed.strip())
        attempted = int(attempted.strip())
        rate = landed / attempted if attempted > 0 else 0.0
        return landed, attempted, rate
    except Exception:
        return np.nan, np.nan, np.nan

def get_result(winner, fighter_name):
    """
    Return result from one fighter's perspective.

    1   -> fighter won
    0   -> fighter lost
    NaN -> draw / no contest / unknown
    """
    if pd.isna(winner):
        return np.nan

    w = str(winner).strip().upper()
    if w == "" or w in ["DRAW", "NO CONTEST", "NC"]:
        return np.nan

    return 1 if str(winner).strip() == str(fighter_name).strip() else 0

def method_group(m):
    """
    Collapse detailed method labels into broad groups.
    """
    m = "" if pd.isna(m) else str(m).strip().lower()

    if "ko/tko" in m:
        return "ko"
    if "submission" in m:
        return "sub"
    if "decision" in m:
        return "dec"

    return "other"

def derandomise_fighter_order(df_raw, seed=42):
    """
    Randomly swats FIghter1/Fighter2 (and all f1_/f2_ columns)
    for 50% of rows so that 'Winner' is no longer correlated with
    which slot a fighter happend to occupy
    """
    df = df_raw.copy()
    rng = np.random.default_rng(seed)
    swap = rng.random(len(df)) < 0.5

    # find every column has a "1" suffix and "2" suffix
    col_pairs = []
    for col in df.columns:
        if col.endswith("1") and col[:-1] + "2" in df.columns:
            col_pairs.append((col, col[:-1] + "2"))
        elif col.startswith("f1_") and  "f2_" + col[3:] in df.columns:
            col_pairs.append((col, "f2_" + col[3:]))
        elif col == "Fighter1":
            col_pairs.append((col, "Fighter2"))

    f1_cols = [a for a, b in col_pairs]
    f2_cols = [b for a,b in col_pairs]


    df.loc[swap, f1_cols + f2_cols] = df.loc[swap, f2_cols + f1_cols].values

    return df


# ELo helpers

def elo_expected(r_a, r_b, scale=400):
    """
    standard elo expected score formula
    """
    return 1.0 / (1.0 + 10 ** (-(r_a - r_b)/ scale))

def build_elo_features(df, base_elo=1500, k=16, scale=400):
    """
        Compute Elo ratings over time and add them to the fight-level dataframe.

        Adds:
            elo_pre_f1
            elo_pre_f2
            elo_post_f1
            elo_post_f2
            elo_diff_pre
            s1
        """
    d = df.sort_values("fight_index").reset_index(drop=True).copy()

    elo = {}
    pre1, pre2, post1, post2, s1_list = [], [], [], [], []

    for _,row in d.iterrows():
        f1 = str(row["Fighter1"]).strip()
        f2 = str(row["Fighter2"]).strip()
        winner = "" if pd.isna(row["Winner"]) else str(row["Winner"]).strip()

        r1 = elo.get(f1, base_elo);
        r2 = elo.get(f2, base_elo);

        pre1.append(r1)
        pre2.append(r2)

        w_up = winner.upper()

        if w_up in ["NO CONTEST", "NC", ""]:
            s1 = np.nan
            s1_list.append(s1)
            post1.append(r1)
            post2.append(r2)
            continue

        if w_up == "DRAW":
            s1 = 0.5
        elif winner == f1:
            s1 = 1.0
        elif winner == f2:
            s1 = 0.0
        else:
            s1 = np.nan

        s1_list.append(s1)

        if np.isnan(s1):
            post1.append(r1)
            post2.append(r2)
            continue

        e1 = elo_expected(r1, r2, scale=scale)

        r1_new = r1 + k * (s1 - e1)
        r2_new = r2 + k * ((1.0 - s1) - (1.0 - e1))

        elo[f1] = r1_new
        elo[f2] = r2_new

        post1.append(r1_new)
        post2.append(r2_new)

    d["elo_pre_f1"] = pre1
    d["elo_pre_f2"] = pre2
    d["elo_post_f1"] = post1
    d["elo_post_f2"] = post2
    d["elo_diff_pre"] = d["elo_pre_f1"] - d["elo_pre_f2"]
    d["s1"] = s1_list

    return d

# Fight level cleaning
def clean_fight_level(df_raw):
    """
    clean the raw enriched fight level dataframe

    Expected columns include:
        Event, Fighter2, Fighter2, Winner, Weightclass, Method,
        Round, Time, event_date,
        f1_dob, f2_dob,
        f1_height, f2_height,
        f1_reach, f2_read,
        f1_stance, f2_stance

    Returns:
        cleaned fight-level dataframe
    """
    df = df_raw.copy()

    #convert time and round into fight duration
    df["round_time_seconds"] = df["Time"].apply(mmss_to_seconds)
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce");
    df["total_fight_seconds"] = (df["Round"] - 1) * 5 * 60 + df["round_time_seconds"]

    #convert date columns
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["f1_dob"] = pd.to_datetime(df["f1_dob"], errors="coerce")
    df["f2_dob"] = pd.to_datetime(df["f2_dob"], errors="coerce")

    # age features
    df["f1_age"] = (df["event_date"] - df["f1_dob"]).dt.days / 365.25
    df["f2_age"] = (df["event_date"] - df["f2_dob"]).dt.days / 365.25
    df["age_diff"] = df["f1_age"] - df["f2_age"]

    # height features
    df["f1_height_in"] = df["f1_height"].apply(height_to_inches)
    df["f2_height_in"] = df["f2_height"].apply(height_to_inches)
    df["height_diff"] = df["f1_height_in"] - df["f2_height_in"]

    # Reach features
    df["f1_reach_in"] = df["f1_reach"].apply(reach_to_inches)
    df["f2_reach_in"] = df["f2_reach"].apply(reach_to_inches)
    df["reach_diff"] = df["f1_reach_in"] - df["f2_reach_in"]

    # stance cleanup
    df["f1_stance_clean"] = df["f1_stance"].apply(clean_stance)
    df["f2_stance_clean"] = df["f2_stance"].apply(clean_stance)

    # sort from oldest to newest
    df = df.sort_values("event_date").reset_index(drop=True)

    # chronological fight index
    df["fight_index"] = np.arange(len(df))

    return df

# =================
# Fight level --> fighter history
# =================

def build_fighter_history(df):
    """
    Convert each fight into two fighter-perspective rows.

    Returns a dataframe where each row represents:
        one fighter, one fight, one opponent
        Done so that model can learn fighter level patterns
    """
    rows = []

    for _, r in df.iterrows():
        f1 = r["Fighter1"]
        f2 = r["Fighter2"]
        m = r["Method"]

        rows.append({
            "fight_index": r["fight_index"],
            "event_date": r["event_date"],
            "event": r["Event"],
            "weightclass": r["Weightclass"],
            "method": m,
            "fighter": f1,
            "opponent": f2,
            "result": get_result(r["Winner"], f1),
            "total_fight_seconds": r["total_fight_seconds"],
            "kd": r["KD1"],
            "sig_str": r["SIG_STR1"],
            "td": r["TD1"],
            "sub_att": r["SUB_ATT1"],
            "rev": r["REV1"],
            "ctrl": r["CTRL1"],
            "stance": r["f1_stance_clean"],
            "opp_stance": r["f2_stance_clean"],
            "own_elo_pre": r["elo_pre_f1"],
            "opp_elo_pre": r["elo_pre_f2"],
        })

        rows.append({
            "fight_index": r["fight_index"],
            "event_date": r["event_date"],
            "event": r["Event"],
            "weightclass": r["Weightclass"],
            "method": m,
            "fighter": f2,
            "opponent": f1,
            "result": get_result(r["Winner"], f2),
            "total_fight_seconds": r["total_fight_seconds"],
            "kd": r["KD2"],
            "sig_str": r["SIG_STR2"],
            "td": r["TD2"],
            "sub_att": r["SUB_ATT2"],
            "rev": r["REV2"],
            "ctrl": r["CTRL2"],
            "stance": r["f2_stance_clean"],
            "opp_stance": r["f1_stance_clean"],
            "own_elo_pre": r["elo_pre_f2"],
            "opp_elo_pre": r["elo_pre_f1"],
        })

    fighter_hist = pd.DataFrame(rows)
    fighter_hist = fighter_hist.sort_values(["fighter", "fight_index"]).reset_index(drop=True)

    # parse significant strikes
    fighter_hist[["sig_landed", "sig_attempted", "sig_rate"]] = (
        fighter_hist["sig_str"].apply(parse_of).apply(pd.Series)
    )

    # Parse takedowns
    fighter_hist[["td_landed", "td_attempted", "td_rate"]] = (
        fighter_hist["td"].apply(parse_of).apply(pd.Series)
    )

    # Other derived numeric features
    fighter_hist["ctrl_seconds"] = fighter_hist["ctrl"].apply(mmss_to_seconds)
    fighter_hist["kd"] = pd.to_numeric(fighter_hist["kd"], errors="coerce")

    # Per-second / proportion stats
    fighter_hist["sig_per_sec"] = fighter_hist["sig_landed"] / fighter_hist["total_fight_seconds"]
    fighter_hist["ctrl_pct"] = fighter_hist["ctrl_seconds"] / fighter_hist["total_fight_seconds"]
    fighter_hist["kd_per_sec"] = fighter_hist["kd"] / fighter_hist["total_fight_seconds"]

    # Method group features
    fighter_hist["method_group"] = fighter_hist["method"].apply(method_group)
    fighter_hist["win_ko"] = (
        (fighter_hist["result"] == 1) & (fighter_hist["method_group"] == "ko")
    ).astype(int)
    fighter_hist["win_sub"] = (
        (fighter_hist["result"] == 1) & (fighter_hist["method_group"] == "sub")
    ).astype(int)

    return fighter_hist

# Rolling pre-fight features

def add_rolling_features(fighter_hist, window=5):
    """
    Add rolling pre-fight features using only prior fights.

    shift(1) is critical because it prevents the current
    fight from leaking into its own features
    """
    d = fighter_hist.copy()
    d = d.sort_values(["fighter", "fight_index"]).reset_index(drop=True)

    rolling_cols = ["sig_per_sec", "sig_rate", "ctrl_pct", "td_rate"]

    for col in rolling_cols:
        d[f"{col}_avg"] = (
            d.groupby("fighter")[col]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
        )
    d["win_rate_avg"] = (
        d.groupby("fighter")["result"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=3).mean())
    )
    # how many prior fights does this fighter have, at this row, period (regardless of NaN)?
    # d["prior_fight_count"] = d.groupby("fighter").cumcount()

    # insufficient_history = d["prior_fight_count"] < 5
    # still_missing_with_history = d["win_rate_avg"].isna() & (~insufficient_history)

    # print("Missing due to <5 career fights so far:", insufficient_history.mean())
    # print("Missing despite having 5+ fights (draw/NC contamination):", still_missing_with_history.mean())

    d["kd_per_sec_avg"] = (
        d.groupby("fighter")["kd_per_sec"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
    )

    d["opp_elo_avg_5"] = (
        d.groupby("fighter")["opp_elo_pre"]
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=5).mean())
    )

    d["opp_elo_avg_3"] = (
        d.groupby("fighter")["opp_elo_pre"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=3).mean())
    )

    d["ko_win_rate_avg"] = (
        d.groupby("fighter")["win_ko"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
    )

    d["sub_win_rate_avg"] = (
        d.groupby("fighter")["win_sub"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
    )


    return d

# Match-up dataset creation
BASE_FEATURE_COLS = [
    "sig_per_sec_avg",
    "sig_rate_avg",
    "ctrl_pct_avg",
    "td_rate_avg",
    "win_rate_avg",
    "kd_per_sec_avg",
    # "exp_prior",
    # "exp_div_prior",
    "ko_win_rate_avg",
    "sub_win_rate_avg",
    # "win_streak_prior",
    # "loss_streak_prior",
]

FINAL_FEATURE_COLS = [f"{c}_diff" for c in BASE_FEATURE_COLS] + [
    "elo_diff_pre",
    "age_diff",
    "height_diff",
    "reach_diff",
]

def build_matchup_dataset(df_fight_level, fighter_hist_rolled):
    """
    Build the final matchup-level dataset for modeling.

    Returns:
        paired, X, y, feature_cols
    """
    # print(fighter_hist_rolled.shape)
    model_df = fighter_hist_rolled.dropna(subset=BASE_FEATURE_COLS + ["result"]).copy()
    model_df["fight_id"] = model_df["fight_index"]
    # print(model_df.shape)

    # keep only fights where both fighters have valid rows
    counts = model_df.groupby("fight_id").size()
    valid_fights = counts[counts == 2].index
    model_df = model_df[model_df["fight_id"].isin(valid_fights)].copy()
    # print(model_df.shape)

    # self-merge to pair each fighter with the other side of the same fight
    paired = model_df.merge(model_df, on="fight_id", suffixes=("_f", "_o"))
    paired = paired[paired["fighter_f"] != paired["fighter_o"]].reset_index(drop=True)
    # print("paired shape:", paired.shape)



    # Directional fight-level differences
    fight_level_long = pd.concat([
        df_fight_level[
            ["fight_index", "Fighter1", "Fighter2", "age_diff", "height_diff", "reach_diff", "elo_diff_pre"]
        ].rename(columns={
            "fight_index": "fight_id",
            "Fighter1": "fighter_f",
            "Fighter2": "fighter_o"
        }),

        df_fight_level[
            ["fight_index", "Fighter2", "Fighter1", "age_diff", "height_diff", "reach_diff", "elo_diff_pre"]
        ].rename(columns={
            "fight_index": "fight_id",
            "Fighter2": "fighter_f",
            "Fighter1": "fighter_o"
        }).assign(
            age_diff=lambda x: -x["age_diff"],
            height_diff=lambda x: -x["height_diff"],
            reach_diff=lambda x: -x["reach_diff"],
            elo_diff_pre=lambda x: -x["elo_diff_pre"]
        )
    ], ignore_index=True)

    paired = paired.merge(
        fight_level_long,
        on=["fight_id", "fighter_f", "fighter_o"],
        how="left"
    )

    #fighter minus opponent feature difference
    for col in BASE_FEATURE_COLS:
        paired[f"{col}_diff"] = paired[f"{col}_f"] - paired[f"{col}_o"]

    X = paired[FINAL_FEATURE_COLS].copy()
    y = paired["result_f"].copy()

    return paired, X, y, FINAL_FEATURE_COLS


def run_preprocessor(df_raw, base_elo=1500, elo_k=16, rolling_window=5):
    """
    Full preprocessing pipeline.

    Steps:
        1. Clean fight-level data
        2. Add Elo features
        3. Build fighter history
        4. Add rolling pre-fight features
        5. Build matchup dataset

    Returns a dictionary of all useful intermediate outputs.
    """
    df_raw = derandomise_fighter_order(df_raw)
    # print(df_raw["Winner"].eq(df_raw["Fighter1"]).mean())
    df_fight = clean_fight_level(df_raw);
    df_fight = build_elo_features(df_fight, base_elo=base_elo, k=elo_k)
    fighter_hist = build_fighter_history(df_fight)
    fighter_hist_rolled = add_rolling_features(fighter_hist, window=rolling_window)
    paired, X, y, feature_cols = build_matchup_dataset(df_fight, fighter_hist_rolled)
    # print("paired shape",paired.shape)

    assert (paired["opponent_f"] == paired["fighter_o"]).all()
    return {
            "df_fight": df_fight,
            "fighter_hist": fighter_hist,
            "fighter_hist_rolled": fighter_hist_rolled,
            "paired": paired,
            "X": X,
            "y": y,
            "feature_cols": feature_cols,
        }

BASE_ELO = 1500
ELO_K = 16
ROLLING_WINDOW = 5

def main():
    df = pd.read_csv("ufc-dataset.csv")
    print("Loaded rows:", len(df))
    print("loaded columns:", len(df.columns))

    # print(df["Winner"].eq(df["Fighter1"]).mean())

    # run preprocessing
    bundle = run_preprocessor(
            df_raw=df,
            base_elo=BASE_ELO,
            elo_k=ELO_K,
            rolling_window=ROLLING_WINDOW,
        )


if __name__ == "__main__":
    main()
