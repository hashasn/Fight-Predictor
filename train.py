import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from preprocessor import run_preprocessor

BASE_ELO = 1500
ELO_K = 16
ROLLING_WINDOW = 5
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 60
PATIENCE = 8
EARLY_STOP_EPS = 1e-4
RANDOM_SEED = 42


# Train/test split
def chronological_split_3way(paired, feature_cols, test_size=0.2, val_size=0.2, output_dir="."):
    """
    Split by fight_id chronologically into:
        train -> oldest fights
        val   -> middle fights
        test  -> newest fights

    val_size is applied to the pre-test portion only.
    """
    sorted_fights = sorted(paired["fight_id"].unique())  #one row. oldest first, newest last
    n_fights = len(sorted_fights)

    test_start = int(n_fights * (1 - test_size))
    pretest_fights = sorted_fights[:test_start]
    test_fights = sorted_fights[test_start:]

    val_start = int(len(pretest_fights) * (1 - val_size))
    train_fights = pretest_fights[:val_start]
    val_fights = pretest_fights[val_start:]

    train_df = paired[paired["fight_id"].isin(train_fights)].copy()
    val_df = paired[paired["fight_id"].isin(val_fights)].copy()
    test_df = paired[paired["fight_id"].isin(test_fights)].copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df["result_f"].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df["result_f"].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df["result_f"].copy()
    # print(test_df.columns)


    return (
        train_df, val_df, test_df,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )



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

    paired = bundle["paired"].copy()
    feature_cols = bundle["feature_cols"]

    print("Paired rows:", len(paired))
    print("Feature count:", len(feature_cols))

    # Train/test split

    (train_df, val_df, test_df,
    X_train, y_train, X_val, y_val,
    X_test, y_test) = chronological_split_3way(
        paired =paired,
        feature_cols=feature_cols,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
    )


    print("Train rows:", X_train.shape)
    print("Val rows:", X_val.shape)
    print("Test rows:", X_test.shape)

    print("Unique train fights:", train_df["fight_id"].nunique())
    print("Unique val fights:", val_df["fight_id"].nunique())
    print("Unique test fights:", test_df["fight_id"].nunique())

    train_val_overlap = set(train_df["fight_id"]) & set(val_df["fight_id"])
    train_test_overlap = set(train_df["fight_id"]) & set(test_df["fight_id"])
    val_test_overlap = set(val_df["fight_id"]) & set(test_df["fight_id"])

    print("Train/Val overlap:", len(train_val_overlap))
    print("Train/Test overlap:", len(train_test_overlap))
    print("Val/Test overlap:", len(val_test_overlap))



if __name__ == "__main__":
    main()
