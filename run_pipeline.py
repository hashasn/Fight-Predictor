import json
from pathlib import Path

import pandas as pd

from event_scraper import get_completed_events, get_upcoming_events, scrape_event
from preprocessor import run_preprocessor
import train as train_module
import predict as predict_module

CSV_PATH = Path("ufc-dataset.csv")
CHECKPOINT_PATH = Path("checkpoint.json")
STORE_PATH = Path("predictions_store.json")


# state I/O

def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "checkpoint.json not found. Create it manually first, e.g.:\n"
            '  {"last_resolved_event": "UFC Fight Night: Moreno vs. Kavanagh", '
            '"last_resolved_date": "2026-02-07"}'
        )
    return json.loads(CHECKPOINT_PATH.read_text())

def save_checkpoint(checkpoint):
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))


def load_store():
    if not STORE_PATH.exists():
        return {"upcoming": [], "archive": []}
    return json.loads(STORE_PATH.read_text())


def save_store(store):
    STORE_PATH.write_text(json.dumps(store, indent=2, default=str))


# csv / bundle

def append_rows_to_csv(rows):
    """Union-safe append: doesn't assume the scraper's columns exactly match
    the existing CSV's. Warns if the column sets differ so mismatches are
    visible instead of silently reshaping your dataset."""
    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if CSV_PATH.exists():
        existing = pd.read_csv(CSV_PATH)
        missing_in_new = set(existing.columns) - set(new_df.columns)
        extra_in_new = set(new_df.columns) - set(existing.columns)
        if missing_in_new:
            print(f"  WARNING: new rows are missing columns the CSV already has: {missing_in_new}")
        if extra_in_new:
            print(f"  WARNING: new rows have columns not in the existing CSV: {extra_in_new}")
        combined = pd.concat([new_df, existing], ignore_index=True, sort=False)
    else:
        combined = new_df

    combined.to_csv(CSV_PATH, index=False)
    print(f"  Appended {len(new_df)} rows to {CSV_PATH} (total now {len(combined)})")


def build_bundle():
    """Rebuild the fighter-history bundle from whatever's currently in the
    CSV. Must be called fresh before predicting each event, since it needs
    to reflect any retrain that happened since the last bundle was built."""
    df = pd.read_csv(CSV_PATH)
    return run_preprocessor(
        df_raw=df,
        base_elo=train_module.BASE_ELO,
        elo_k=train_module.ELO_K,
        rolling_window=train_module.ROLLING_WINDOW,
    )


#  predicting

def predict_event_fights(event_dict, fighter_pairs,bundle):
    results = []

    for fighter_a, fighter_b in fighter_pairs:
        try:
            pred = predict_module.predict_matchup(
                fighter_a, fighter_b, bundle, fight_date=event_dict["date_dt"]
            )
            pred["event"] = event_dict["name"]
            pred["date"] = event_dict["date"]
            results.append(pred)

        except ValueError as e:
            print(f"    Skipping {fighter_a} vs {fighter_b}: {e}")
    return results

# pull out fighter pairs from list of fight rows
def fighter_pairs_from_rows(rows):
    pairs = []
    for row in rows:
        pair = (row["Fighter1"], row["Fighter2"])
        pairs.append(pair)
    return pairs


# core per-event step
def process_event(event_dict, bundle, predicted=None):
    """
    Scrape one event once. If `predicted` is None, this is a backlog event:
    predict it fresh using the CURRENT (pre-retrain) model, using the same
    scrape to get both the matchups and the actual results. If `predicted`
    is given, this is a previously-pending event -- just resolve it against
    the real results, no new prediction needed.

    Appends to the CSV and retrains regardless. Returns (archive_entries,
    metrics), or None if nothing could be scraped (event page empty/broken).
    """
    rows = scrape_event(event_dict)
    if not rows:
        print(f"  Nothing scraped for {event_dict['name']}.")
        return None

    if predicted is None:
        pairs = fighter_pairs_from_rows(rows)
        predicted = predict_event_fights(event_dict, pairs, bundle)

    by_pair = {(r["Fighter1"], r["Fighter2"]): r for r in rows}
    archive_entries = []
    for p in predicted:
        row = by_pair.get((p["fighter_a"], p["fighter_b"]))
        if not row:
            print(f"  No matching result row for {p['fighter_a']} vs {p['fighter_b']}, skipping.")
            continue


        predicted_winner = p["fighter_a"] if p["fighter_a_win_prob"] >= 0.5 else p["fighter_b"]
        actual_winner = str(row["Winner"]).strip()



        archive_entries.append({
            **p,
            "predicted_winner": predicted_winner,
            "actual_winner": actual_winner,
            "method": row["Method"],
            "correct": (
                None if actual_winner.upper() in ("DRAW", "NO CONTEST")
                else predicted_winner.strip() == actual_winner
            ),
        })

    append_rows_to_csv(rows)

    print("  Retraining...")
    metrics = train_module.train_and_save(csv_path=str(CSV_PATH))
    predict_module.reload_artifacts()

    return archive_entries, metrics


#  main

def main():
    checkpoint = load_checkpoint()
    store = load_store()
    last_date = pd.Timestamp(checkpoint["last_resolved_date"])

    completed_sorted = sorted(get_completed_events(), key=lambda e: e["date_dt"])

    # --- Step 1: resolve a previously pending prediction, if it's now completed ---
    if store["upcoming"]:
        pending_event_name = store["upcoming"][0]["event"]
        match = next((e for e in completed_sorted if e["name"] == pending_event_name), None)

        if match:
            print(f"Resolving previously pending event: {pending_event_name}")
            bundle = build_bundle()
            result = process_event(match, bundle, predicted=store["upcoming"])
            if result:
                archive_entries, metrics = result
                store["archive"].extend(archive_entries)
                store["upcoming"] = []
                checkpoint = {"last_resolved_event": match["name"], "last_resolved_date": match["date"]}
                save_checkpoint(checkpoint)
                save_store(store)
                last_date = match["date_dt"]
                print(f"  Resolved. val_auc={metrics.get('val_auc'):.3f}")
        else:
            print(f"Still waiting on: {pending_event_name} (not yet completed)")

    # --- Step 2: walk forward through any remaining backlog of completed events ---
    while True:
        next_completed = next((e for e in completed_sorted if e["date_dt"] > last_date), None)
        if not next_completed:
            break

        print(f"Backlog event: {next_completed['name']} ({next_completed['date']})")
        bundle = build_bundle()
        result = process_event(next_completed, bundle, predicted=None)

        if not result:
            print("  Skipping this event (nothing scraped) to avoid getting stuck.")
            last_date = next_completed["date_dt"]
            continue

        archive_entries, metrics = result
        store["archive"].extend(archive_entries)
        checkpoint = {"last_resolved_event": next_completed["name"], "last_resolved_date": next_completed["date"]}
        save_checkpoint(checkpoint)
        save_store(store)
        last_date = next_completed["date_dt"]
        print(f"  Done. val_auc={metrics.get('val_auc'):.3f}")

    # --- Step 3: caught up -- predict the next real upcoming event, if not already pending ---
    store = load_store()
    if store["upcoming"]:
        print(f"Prediction already pending for: {store['upcoming'][0]['event']}. Nothing more to do.")
        return

    upcoming_sorted = sorted(get_upcoming_events(), key=lambda e: e["date_dt"])
    next_upcoming = next((e for e in upcoming_sorted if e["date_dt"] > last_date), None)

    if not next_upcoming:
        print("No new upcoming event found. Nothing more to do.")
        return

    print(f"Predicting next upcoming event: {next_upcoming['name']} ({next_upcoming['date']})")
    bundle = build_bundle()
    rows = scrape_event(next_upcoming)

    if not rows:
        print("  Nothing scraped for the upcoming card -- see KNOWN GAPS at top of this file.")
        return

    pairs = fighter_pairs_from_rows(rows)
    predicted = predict_event_fights(next_upcoming, pairs, bundle)
    store["upcoming"] = predicted
    save_store(store)
    print(f"  Stored {len(predicted)} predictions for {next_upcoming['name']}.")


if __name__ == "__main__":
    main()
