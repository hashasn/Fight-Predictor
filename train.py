import numpy as np
import pandas as pd
from preprocessor import run_preprocessor

BASE_ELO = 1500
ELO_K = 16
ROLLING_WINDOW = 5

def main():
    #load dataset
    df = pd.read_csv("ufc-dataset.csv")
    print("Loaded rows:", len(df))
    # print("loaded columns:", len(df.columns))
    # print("columns: ", df.dtypes)

    # run preprocessing
    bundle = run_preprocessor(
            df_raw=df,
            base_elo=BASE_ELO,
            elo_k=ELO_K,
            rolling_window=ROLLING_WINDOW,
        )

    feature_cols = bundle["feature_cols"]
    print("Feature cols ", len(feature_cols))


if __name__ == "__main__":
    main()
