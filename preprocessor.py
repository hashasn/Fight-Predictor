import numpy as np
import pandas as pd
import re

# helpers



def mmss_to_seconds2(value):
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
    df["round_time_seconds"] = df["Time"].apply(mmss_to_seconds2)
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
    print(df_raw["Winner"].eq(df_raw["Fighter1"]).mean())

    df_fight = clean_fight_level(df_raw);

    return {"df_fight": df_fight}

BASE_ELO = 1500
ELO_K = 16
ROLLING_WINDOW = 5

def main():
    df = pd.read_csv("ufc-dataset.csv")
    print("Loaded rows:", len(df))
    print("loaded columns:", len(df.columns))

    print(df["Winner"].eq(df["Fighter1"]).mean())

    # run preprocessing
    bundle = run_preprocessor(
            df_raw=df,
            base_elo=BASE_ELO,
            elo_k=ELO_K,
            rolling_window=ROLLING_WINDOW,
        )

if __name__ == "__main__":
    main()
