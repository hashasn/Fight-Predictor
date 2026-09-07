import pandas as pd
import numpy as np

from preprocessor import run_preprocessor
from predict import predict_matchup

def main():
    df = pd.read_csv("ufc-dataset.csv")
    bundle = run_preprocessor(df, base_elo=1500, elo_k=16, rolling_window=5)
    # result = predict_matchup("Brandon Moreno","Lone'er Kavanagh", bundle)
    # print(result)
    fighters_to_check = [
        "Dan Hooker", "Salahdine Parnasse",
        "Michael Page", "Nursulton Ruziboev",
        "Ryan Spann", "Mario Pinto",
        "Modestas Bukauskas", "Oumar Sy",
        "Nathaniel Wood", "Pavel Andrusca",
    ]

    # for name in fighters_to_check:
    #     count = bundle["fighter_hist_rolled"][bundle["fighter_hist_rolled"]["fighter"] == name].shape[0]
    #     print(f"{name}: {count} fights in dataset")
    #
    upcoming_fights = [
        {"event": "UFC Fight Night: Hooker vs. Parnasse", "fighter_a": "Michael Page", "fighter_b": "Nursulton Ruziboev", "date": "2026-09-05"},
        {"event": "UFC Fight Night: Hooker vs. Parnasse", "fighter_a": "Ryan Spann", "fighter_b": "Mario Pinto", "date": "2026-09-05"},
        {"event": "UFC Fight Night: Hooker vs. Parnasse", "fighter_a": "Modestas Bukauskas", "fighter_b": "Oumar Sy", "date": "2026-09-05"},
    ]

    results = []
    for fight in upcoming_fights:
        pred = predict_matchup(
            fight["fighter_a"], fight["fighter_b"], bundle,
            fight_date=pd.Timestamp(fight["date"])
        )
        results.append({**fight, **pred})

    import json
    with open("predictions.json", "w") as f:
        json.dump(results,f, indent=2)

    # print(json.dump(results,f,indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
