import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from preprocessor import run_preprocessor
import train as train_module
import predict as predict_module

app = FastAPI()

CSV_PATH = "data/ufc-dataset.csv"
STORE_PATH = Path("data/predictions_store.json")


#allow the react dev server to call this API
app.add_middleware(
    CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
)

# request/response shapes
class MatchupRequest(BaseModel):
    fighter_a: str
    fighter_b: str


class MatchupResponse(BaseModel):
    fighter_a: str
    fighter_b: str
    fighter_a_win_prob: float
    fighter_b_win_prob: float


def build_bundle():
    """
    rebuild the fighter history currently in the csv
    """
    df = pd.read_csv(CSV_PATH)
    bundle = run_preprocessor(
        df_raw=df,
        base_elo=train_module.BASE_ELO,
        elo_k=train_module.ELO_K,
        rolling_window=train_module.ROLLING_WINDOW,
    )
    return bundle


@app.get("/health")
def health():
    return {"status": "ok"}

def load_store():
    if not STORE_PATH.exists():
        return {"upcoming": [], "archive": []}
    return json.loads(STORE_PATH.read_text())

@app.get("/predictions/upcoming")
def get_upcoming():
    store = load_store()
    return store["upcoming"]

@app.get("predictions/archive")
def get_archive():
    store = load_store()
    return store["archive"]

@app.post("/predict", response_model=MatchupResponse)
def predict(request: MatchupRequest):
    predict_module.reload_artifacts()

    bundle = build_bundle()

    try:
        result = predict_module.predict_matchup(
            request.fighter_a, request.fighter_b, bundle
        )
    except ValueError as e:
        # raised when a fighter has no history in the dataset
        raise HTTPException(status_code=404, detail=str(e))

    return result
