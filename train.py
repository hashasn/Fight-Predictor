import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from preprocessor import run_preprocessor

import json
import joblib
from pathlib import Path
from datetime import datetime, timezone


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

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

# def train_logistic(X_train, y_train, X_val, y_val):


#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)

#     train_pred = model.predict(X_train)
#     print("Train accuracy: ", accuracy_score(y_train, train_pred))

#     val_pred = model.predict(X_val)
#     val_prob = model.predict_proba(X_val)[:, 1]

#     print("accuracy: ", accuracy_score(y_val, val_pred))
#     print("Val AUC: ", roc_auc_score(y_val, val_prob))
#     # sanity check: naive baselines
#     elo_auc = roc_auc_score(y_val, X_val["elo_diff_pre"])
#     print("Elo-only AUC:", elo_auc)

#     print(confusion_matrix(y_val, val_pred))
#     print(classification_report(y_val, val_pred))

#     return model

# def train_hgb(X_train, y_train, X_val, y_val):
#     model = HistGradientBoostingClassifier(
#         max_iter=100,
#         max_depth=3,
#         learning_rate=0.05,
#         random_state=RANDOM_SEED
#     )
#     model.fit(X_train, y_train)

#     train_pred = model.predict(X_train)
#     print("Train accuracy: ", accuracy_score(y_train, train_pred))

#     val_pred = model.predict(X_val)
#     val_prob = model.predict_proba(X_val)[:, 1]

#     print("HGB accuracy: ", accuracy_score(y_val, val_pred))
#     print("HGB Val AUC: ", roc_auc_score(y_val, val_prob))

#     print(confusion_matrix(y_val, val_pred))
#     print(classification_report(y_val, val_pred))

#     return model

# def train_rf(X_train, y_train, X_val, y_val):
#     model = RandomForestClassifier(
#         n_estimators=300,
#         max_depth=5,          # keep shallow to limit overfitting on ~3000 rows
#         min_samples_leaf=10,  # require some minimum data per leaf, same reason
#         random_state=RANDOM_SEED
#     )
#     model.fit(X_train, y_train)
#     train_pred = model.predict(X_train)
#     print("Train accuracy: ", accuracy_score(y_train, train_pred))

#     val_pred = model.predict(X_val)
#     val_prob = model.predict_proba(X_val)[:, 1]

#     print("RF accuracy: ", accuracy_score(y_val, val_pred))
#     print("RF Val AUC: ", roc_auc_score(y_val, val_prob))
#     print(confusion_matrix(y_val, val_pred))

#     return model

class FightNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).numpy()
        pred = (probs >= 0.5).astype(int)

    return pred, probs

def train_nn(X_train, y_train, X_val, y_val, X_test, y_test,  feature_cols):
    torch.manual_seed(RANDOM_SEED)

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = FightNet(len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        # print(f"Epoch {epoch+1}/{EPOCHS} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss - EARLY_STOP_EPS:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    # final eval
    val_pred, val_probs = evaluate_model(model, X_val_t, y_val_t)
    # model.eval()
    # with torch.no_grad():
    #     val_logits = model(X_val_t)
    #     val_probs = torch.sigmoid(val_logits).numpy()
    #     val_pred = (val_probs >= 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_probs)
    print("NN val accuracy:", val_acc)
    print("NN Val AUC:", val_auc)
    print(confusion_matrix(y_val, val_pred))

    test_pred, test_probs = evaluate_model(model, X_test_t, y_test_t)
    test_acc = accuracy_score(y_test, test_pred)
    test_auc =  roc_auc_score(y_test, test_probs)
    print("FINAL TEST NN accuracy:",test_acc )
    print("FINAL TEST NN AUC:", test_auc)
    print(confusion_matrix(y_test, test_pred))
    history = {
        "val_pred": val_pred,
        "val_probs": val_probs,
        "val_accuracy": val_acc,
        "val_AUC": val_auc,
        "test_pred": test_pred,
        "test_probs": test_probs,
        "test_accuracy": test_acc,
        "test AUC": test_auc
    }

    return model, history


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
    print("NaNs in X_train before imputing:", X_train.isna().sum().sum())



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

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )



    # print("\n--- Logistic Regression ---")
    # lr_model = train_logistic(X_train_scaled, y_train, X_val_scaled, y_val)

    # test_pred = lr_model.predict(X_test_scaled)
    # test_prob = lr_model.predict(X_test_scaled)[:,1]

    # print("FINAL TEST accuracy:", accuracy_score(y_test, test_pred))
    # print("FINAL TEST AUC:", roc_auc_score(y_test, test_prob))
    # print(confusion_matrix(y_test, test_pred))

    # print("\n--- Random Forest ---")
    # rf_model = train_rf(X_train_scaled, y_train, X_val_scaled, y_val)

    # print("\n--- HistGradientBoosting ---")
    # hgb_model = train_hgb(X_train_scaled, y_train, X_val_scaled, y_val)

    print("\n--- Neural Network ---")
    nn_model, history = train_nn(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_cols)


    # 1. Model weights
    torch.save(nn_model.state_dict(), ARTIFACTS_DIR / "fightnet_state.pt")

    # 2. Preprocessing objects — required to transform any future matchup the same way
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")

    # 3. Feature list and architecture info — needed to reconstruct FightNet before loading weights
    with open(ARTIFACTS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # 4. Metrics — your honest, final numbers
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "FightNet (NN)",
        "base_elo": BASE_ELO,
        "elo_k": ELO_K,
        "rolling_window": ROLLING_WINDOW,
        "n_features": len(feature_cols),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "val_accuracy": accuracy_score(y_val, history["val_pred"]),
        "val_auc": roc_auc_score(y_val, history["val_probs"]),
        "test_accuracy": accuracy_score(y_test, history["test_pred"]),
        "test_auc": roc_auc_score(y_test, history["test_probs"]),
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model + artifacts to", ARTIFACTS_DIR.resolve())
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
