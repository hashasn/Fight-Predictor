import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
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


# baseline model

def train_logistic(X_train, y_train, X_val, y_val):


    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    print("accuracy: ", accuracy_score(y_val, val_pred))
    print("Val AUC: ", roc_auc_score(y_val, val_prob))
    # sanity check: naive baselines
    elo_auc = roc_auc_score(y_val, X_val["elo_diff_pre"])
    print("Elo-only AUC:", elo_auc)

    print(confusion_matrix(y_val, val_pred))
    print(classification_report(y_val, val_pred))

    return model




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


    # print("Train rows:", X_train.shape)
    # print("Val rows:", X_val.shape)
    # print("Test rows:", X_test.shape)

    # print("Unique train fights:", train_df["fight_id"].nunique())
    # print("Unique val fights:", val_df["fight_id"].nunique())
    # print("Unique test fights:", test_df["fight_id"].nunique())

    # train_val_overlap = set(train_df["fight_id"]) & set(val_df["fight_id"])
    # train_test_overlap = set(train_df["fight_id"]) & set(test_df["fight_id"])
    # val_test_overlap = set(val_df["fight_id"]) & set(test_df["fight_id"])

    # print("Train/Val overlap:", len(train_val_overlap))
    # print("Train/Test overlap:", len(train_test_overlap))
    # print("Val/Test overlap:", len(val_test_overlap))

    # print(X_train.isna().sum())

    train_df = train_df.dropna(subset=feature_cols)
    val_df = val_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)

    X_train = train_df[feature_cols].copy()
    y_train = train_df["result_f"].copy()
    X_val = val_df[feature_cols].copy()
    y_val = val_df["result_f"].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["result_f"].copy()

    # print(paired[paired["reach_diff"].isna()][["fighter_f","fighter_o" ]])
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_cols,
        index=X_val.index
    )

    model = train_logistic(X_train_scaled, y_train, X_val_scaled, y_val)

    # feature importance
    coefs = pd.Series(model.coef_[0], index=feature_cols).sort_values()
    print(coefs)



if __name__ == "__main__":
    main()
