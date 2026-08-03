# NYC Taxi Trip Duration Estimator

Predicts taxi trip duration in New York City from pickup/dropoff coordinates,
pickup time, and passenger count — the same core problem behind the ETA feature
in apps like Uber or Ola. Built end-to-end: data cleaning, feature engineering,
model comparison across 4 algorithms, two ensembling approaches, and a deployed
Streamlit demo.

**Final model:** Tuned LightGBM Regressor — RMSLE **0.3644** on held-out validation data
(1.43M cleaned trips, from the [Kaggle NYC Taxi Trip Duration dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)).

## Demo

<!-- Add 1-2 screenshots of the Streamlit app here once captured -->
<!-- ![App screenshot](docs/screenshot.png) -->

## Tech Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · Streamlit · joblib

## Project Architecture

```text
nyc-taxi-duration-estimator/
├── data/raw/, data/processed/   # gitignored — see Setup below
├── notebooks/                   # EDA and model experimentation
├── src/
│   ├── data/load.py             # raw data loading + validation
│   ├── features/engineer.py     # SINGLE SOURCE OF TRUTH for feature logic —
│   │                             # used by both training and inference
│   ├── models/train.py          # model training functions, OOF stacking utils
│   ├── evaluation/metrics.py    # RMSLE, RMSE, MAE
│   └── inference/predict.py     # validation -> features -> predict -> format
├── app/streamlit_app.py         # UI only, imports only from src.inference.predict
├── models/                      # serialized final model
└── tests/                       # pytest suite
```

The key design principle: **`src/features/engineer.py` is imported by both the
training pipeline and `src/inference/predict.py`**, so training and live
predictions can never silently drift out of sync (train/serve skew).

## Setup

```bash
git clone <your-repo-url>
cd nyc-taxi-duration-estimator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

1. Download `train.csv` from the [Kaggle competition page](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)
   (requires a free Kaggle account).
2. Place it at `data/raw/train.csv`.
3. Build the processed feature dataset:
```bash
   python -m src.features.engineer
```

### Run the app

```bash
streamlit run app/streamlit_app.py
```

### Run tests

```bash
pytest tests/ -v
```

## Data Cleaning

Starting from 1,458,644 raw trips, four filters removed 2.02% of rows (final:
1,429,225 trips):

| Filter | Rows removed | Reasoning |
|---|---|---|
| Coordinate bounds (NYC box) | 19,461 | GPS errors far outside NYC |
| Passenger count (1-6) | 53 | 0 passengers or above realistic taxi capacity |
| Duration bounds (60s-22h) | 9,878 | Logging errors (multi-day "trips") |
| Implied speed (>100 km/h) | 27 | Physically implausible distance/duration combos |

Full reasoning for each threshold is in `notebooks/01_eda.ipynb` and
`src/features/engineer.py` docstrings.

## Feature Engineering

- **`distance_km`** — Haversine great-circle distance between pickup/dropoff.
  The single most predictive feature (~63-73% of gain-based importance across
  all three tree models).
- **`pickup_hour`, `pickup_dayofweek`, `is_weekend`** — capture traffic-pattern
  variation (median trip duration varies meaningfully by hour, per EDA).

## Model Comparison

Trained on an 80/20 random split (justification: no meaningful drift within
the dataset's 6-month window — see `notebooks/02_model_experiments.ipynb`).

| Model | RMSLE | RMSE | MAE | Train Time |
|---|---:|---:|---:|---:|
| Linear Regression | 0.5086 | 568.87 | 277.21 | 0.98s |
| Random Forest | 0.3747 | 514.80 | 203.88 | 143.86s |
| XGBoost | 0.3734 | 508.54 | 203.36 | 7.58s |
| LightGBM | 0.3694 | 506.72 | 200.01 | 5.11s |
| **LightGBM (tuned)** | **0.3644** | **506.04** | **197.01** | 5.95s |
| Equal-weight blend (RF+XGB+LGBM) | 0.3666 | 505.06 | 198.16 | — |
| Validation-weighted blend | 0.3665 | 505.03 | 198.12 | — |
| Stacking (OOF + Linear meta-learner) | 0.3634 | 504.56 | 197.16 | ~1hr (OOF) |

### Final model: Tuned LightGBM (standalone)

Stacking edged out standalone LightGBM on RMSLE (0.3634 vs 0.3644, a 0.27%
relative improvement) but was *worse* on MAE a mixed, marginal result. Against
that tiny gain:

| | Tuned LightGBM | Stacking ensemble |
|---|---:|---:|
| Model size | 1.05 MB | 167.14 MB |
| Single-prediction latency | 0.96 ms | 36.60 ms |

Random Forest alone accounted for 99% of the stacking bundle's size and 58% of
its latency, despite being the weakest of the three base models on every
accuracy metric. For a rider-facing ETA feature, a sub-millisecond response at
effectively equivalent accuracy is the better production tradeoff so the
project ships standalone tuned LightGBM, not the ensemble.

## Limitations

- No lower-bound speed filter was applied trips with near-zero distance but
  non-trivial duration (a deliberate, discussed tradeoff to avoid also
  filtering out genuine gridlock-traffic trips) remain in the training data.
- Prediction error grows with trip length in absolute terms (from ~120s MAE on
  5-10 minute trips to ~870s on 40+ minute trips) RMSLE was chosen as the
  primary metric specifically because it's relative, not absolute, but very
  long trips remain the hardest case for this model.
- Straight-line (Haversine) distance is used as a proxy for actual driven
  distance no road-network routing is used, which likely under-represents
  trips that require significant detours (bridges, one-way systems, etc).
- Trained on 6 months of 2016 data no evaluation of how the model performs
  on more recent traffic patterns, road layouts, or trip-time distributions.

## Future Improvements

- Road-network-based distance (e.g. OSRM) instead of straight-line Haversine.
- Weather data as an additional feature (rain/snow plausibly affects duration).
- Time-based validation split to explicitly test for temporal drift.

## Acknowledgments

Dataset: [NYC Taxi Trip Duration, Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration/data),
based on data released by the NYC Taxi & Limousine Commission.
