# UFC Fight Predictor

A project I built to see if I could actually predict UFC fight outcomes using machine learning instead of just guessing based on vibes. It's a full-stack app: a React frontend where you can pick two fighters and get a win probability, a FastAPI backend serving the model, and a pipeline that scrapes new fight results and retrains the model automatically as new events happen.

**What it does:**
- **Head-to-head predictor**: pick any two fighters, get a win % breakdown
- **Upcoming card**: see the model's predictions for the next UFC event before it happens
- **Archive**: every past prediction next to what actually happened, so I can't just claim it works; you can see the track record

## How it works

I pull historical fight data and for each fighter work out their "form" going into a fight: rolling averages over their last 5 fights (strikes landed, takedown rate, control time, win rate, finish rate, etc.) plus an Elo rating that updates after every fight, similar to chess ratings. The actual features fed into the model are the *differences* between the two fighters' stats (e.g. `elo_diff_pre`, `reach_diff`) rather than raw numbers.

The model itself is a small neural network in PyTorch. The important bit is how I split the data: chronologically, not randomly. Training on older fights and testing on newer ones, because in real life you'd never have future data available when predicting a fight, so testing any other way would be cheating (and would make the accuracy numbers meaningless).

**Current results on held-out test data (~1000 fights it never trained on):**
| Metric | Score |
|---|---|
| Accuracy | 63.6% |
| AUC | 0.67 |
| Log loss | 0.65 |

Not amazing, but meaningfully better than a coin flip, and honestly that's about what I expected.

`run_pipeline.py` automates the whole loop: check if a pending prediction can be resolved against a completed event, scrape and add any new results, retrain the model, then generate a prediction for the next upcoming card. It keeps track of where it's up to in `checkpoint.json` so I can just re-run it whenever there's a new event.

## Stack

- **Backend:** FastAPI (Python), REST API for predictions and stored results
- **ML:** PyTorch for the model, scikit-learn for preprocessing, Pandas/NumPy for feature engineering
- **Scraping:** `requests` + BeautifulSoup
- **Frontend:** React (Vite)

## Project structure

```
app.py               FastAPI app, /predict, /predictions/upcoming, /predictions/archive
preprocessor.py     Rolling stats + Elo feature engineering
train.py             Model (FightNet), training loop, chronological train/val/test split
predict.py           Loads the trained model and predicts a single matchup
event_scraper.py     Scrapes completed & upcoming UFC event data
run_pipeline.py      Runs the whole loop: scrape -> retrain -> predict
artifacts/            Saved model weights, preprocessor, feature list, metrics
data/                 Dataset, prediction store, pipeline checkpoint
frontend/             React UI
```

## Running it

**Backend**
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

**Full pipeline**
```bash
python run_pipeline.py
```

## Notes

Not a betting tool. 
I built this to learn how to take a ML ideal all the way from raw data to something usable
