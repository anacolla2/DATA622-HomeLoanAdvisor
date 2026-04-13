# Home Loan Advisor — Flask App

Consumer-facing mortgage risk assessment tool trained on Freddie Mac loan data.

## Project Structure

```
homeloan_flask/
├── app.py                  # Flask app — API + UI
├── build_dataset.py        # Automation pipeline for retraining
├── requirements.txt
├── model_assets/           # Drop your saved model files here
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.json
│   └── state_def_rate.csv
└── templates/
    └── index.html          # Consumer UI
```

## Setup

```bash
pip install -r requirements.txt
```

Copy your four model asset files into `model_assets/`:
- `model.pkl`
- `scaler.pkl`
- `feature_columns.json`
- `state_def_rate.csv`

## Run locally

```bash
python app.py
# → http://localhost:5000
```

## API

```
POST /predict
Content-Type: application/json

{
    "credit_score": 720,
    "dti": 35,
    "ltv": 80,
    "or_interest_rate": 4.5,
    "num_borrowers": 2,
    "prop_state": "NY",
    "first_time_buyer": 0,
    "occupancy": "P",
    "prop_type": "SF",
    "channel": "R",
    "num_units": 1,
    "metro_area": 1,
    "super_conforming": 0,
    "mi_cancel": 0
}
```

Response:
```json
{
    "probability": 0.0823,
    "probability_pct": "8.2%",
    "risk_tier": "moderate",
    "risk_label": "🟡 Moderate Risk",
    "explanation": "Some risk factors are present...",
    "top_factors": [...],
    "state_rate": 0.0512
}
```

## Retrain on new data

```bash
# sample mode (fast, laptop-friendly)
python build_dataset.py --data_dir data/ --sample 50000

# full data (use Google Colab for large datasets)
python build_dataset.py --data_dir data/

# output goes to model_assets/ automatically
```

## Deploy

**Render.com:**
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

**Railway / Fly.io:**
- Same start command: `gunicorn app:app`
