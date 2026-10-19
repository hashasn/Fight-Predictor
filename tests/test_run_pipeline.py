import run_pipeline


# process_event

def make_result_row(fighter1, fighter2, winner, method):
    """
    a scraped fight result row, with only the columns process_event reads
    """
    return {"Fighter1": fighter1, "Fighter2": fighter2, "Winner": winner, "Method": method}


def make_prediction(fighter_a, fighter_b, prob_a):
    """
    a stored prediction for an upcoming fight
    """
    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "fighter_a_win_prob": prob_a,
        "fighter_b_win_prob": round(1 - prob_a, 3),
        "event": "Noche UFC: Silva vs. Delgado",
        "date": "September 12, 2026",
    }


def test_process_event_marks_predictions(monkeypatch):
    """
    each stored prediction should be compared with the real result:
    right pick -> True, wrong pick -> False, draw -> None.
    fights with no result (e.g. cancelled) are skipped
    """
    results = [
        make_result_row("Jean Silva", "Jose Delgado", "Jean Silva", "Submission"),
        make_result_row("Brandon Moreno", "Joseph Morales", "Brandon Moreno", "Decision - Split"),
        make_result_row("Fighter One", "Fighter Two", "DRAW", "Decision - Majority"),
    ]
    predicted = [
        make_prediction("Jean Silva", "Jose Delgado", 0.61),         # picked Silva, Silva won
        make_prediction("Brandon Moreno", "Joseph Morales", 0.40),   # picked Morales, Moreno won
        make_prediction("Fighter One", "Fighter Two", 0.55),         # draw
        make_prediction("Cancelled A", "Cancelled B", 0.70),         # fight never happened
    ]

    # replace scraping, saving and retraining with fakes
    appended = []
    train_calls = []

    monkeypatch.setattr(run_pipeline, "scrape_event", lambda event: results)
    monkeypatch.setattr(run_pipeline, "append_rows_to_csv", lambda rows: appended.append(rows))
    monkeypatch.setattr(run_pipeline.train_module, "train_and_save",
                        lambda csv_path: train_calls.append(csv_path) or {"val_auc": 0.65})
    monkeypatch.setattr(run_pipeline.predict_module, "reload_artifacts", lambda: None)

    event = {"name": "Noche UFC: Silva vs. Delgado", "date": "September 12, 2026", "url": "fake"}
    archive_entries, metrics = run_pipeline.process_event(event, bundle={}, predicted=predicted)

    # cancelled fight is skipped, the other 3 are archived
    assert len(archive_entries) == 3

    silva, moreno, draw = archive_entries

    assert silva["predicted_winner"] == "Jean Silva"
    assert silva["actual_winner"] == "Jean Silva"
    assert silva["method"] == "Submission"
    assert silva["correct"] is True

    assert moreno["predicted_winner"] == "Joseph Morales"
    assert moreno["actual_winner"] == "Brandon Moreno"
    assert moreno["correct"] is False

    # draws don't count as right or wrong
    assert draw["correct"] is None

    # results are saved and the model is retrained once
    assert appended == [results]
    assert len(train_calls) == 1
    assert metrics == {"val_auc": 0.65}


def test_process_event_nothing_scraped(monkeypatch):
    """
    if the event page is empty, nothing should be saved or retrained
    """
    train_calls = []

    monkeypatch.setattr(run_pipeline, "scrape_event", lambda event: [])
    monkeypatch.setattr(run_pipeline.train_module, "train_and_save",
                        lambda csv_path: train_calls.append(csv_path))

    event = {"name": "Empty Event", "date": "September 12, 2026", "url": "fake"}
    result = run_pipeline.process_event(event, bundle={}, predicted=[])

    assert result is None
    assert train_calls == []