# ============================================================================
# Home Loan Advisor — Flask App
# DATA 622 | Spring 2026
# ============================================================================
# USAGE:
#   pip install flask scikit-learn pandas numpy
#   python app.py
#
# API ENDPOINT:
#   POST /predict
#   Body: JSON with user inputs
#   Returns: { probability, risk_tier, risk_label, top_factors }
#
# TO SWAP MODEL:
#   Replace files in model_assets/:
#     model.pkl, scaler.pkl, state_def_rate.csv, feature_columns.json
# ============================================================================

import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load model assets at startup ─────────────────────────────────────────────
with open("model_assets/model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("model_assets/scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

FEATURE_COLS = json.load(open("model_assets/feature_columns.json"))

STATE_RATES = pd.read_csv(
    "model_assets/state_def_rate.csv",
    index_col=0
).squeeze("columns").to_dict()

NATIONAL_AVG_RATE = np.mean(list(STATE_RATES.values()))

print(f"Model loaded — {len(FEATURE_COLS)} features")
print(f"State lookup — {len(STATE_RATES)} states")

# ── Feature medians for reference ────────────────────────────────────────────
MEDIANS = {
    "credit_score":      740,
    "dti":               33,
    "ltv":               75,
    "or_interest_rate":  4.5,
    "num_borrowers":     2,
}

# ── Risk tier logic ──────────────────────────────────────────────────────────
def assign_risk_tier(prob):
    if prob < 0.05:
        return "low",      "🟢 Low Risk",      "Your profile looks strong. Borrowers like you rarely fall behind on payments."
    elif prob < 0.10:
        return "moderate", "🟡 Moderate Risk",  "Some risk factors are present. Consider improving your credit score or reducing debt before committing."
    else:
        return "high",     "🔴 High Risk",      "Multiple risk factors are elevated. We recommend addressing your DTI and credit score, or considering a smaller loan."

# ── Build prediction row from user inputs ────────────────────────────────────
def build_input_row(data):
    """
    Takes user-facing inputs and constructs the full feature vector
    in the exact order the model was trained on.
    """
    row = {col: 0 for col in FEATURE_COLS}

    # ── Numeric features ──────────────────────────────────────────────
    row["credit_score"]     = float(data.get("credit_score", MEDIANS["credit_score"]))
    row["dti"]              = float(data.get("dti", MEDIANS["dti"]))
    row["ltv"]              = float(data.get("ltv", MEDIANS["ltv"]))
    row["or_interest_rate"] = float(data.get("or_interest_rate", MEDIANS["or_interest_rate"]))
    row["num_borrowers"]    = float(data.get("num_borrowers", MEDIANS["num_borrowers"]))

    # ── State default rate lookup ─────────────────────────────────────
    state = data.get("prop_state", "").upper()
    row["state_def_rate"] = STATE_RATES.get(state, NATIONAL_AVG_RATE)

    # ── Binary flags ──────────────────────────────────────────────────
    row["super_conforming"]    = int(data.get("super_conforming", 0))
    row["mi_cancel"]           = int(data.get("mi_cancel", 0))
    row["metro_area"]          = int(data.get("metro_area", 1))
    row["first_time_buyer_Y"]  = int(data.get("first_time_buyer", 0))

    # ── Occupancy (one-hot, base = O) ─────────────────────────────────
    occupancy = data.get("occupancy", "P")
    if occupancy == "P":
        row["occupancy_P"] = 1
    elif occupancy == "S":
        row["occupancy_S"] = 1

    # ── Channel (one-hot, base = B) ───────────────────────────────────
    channel = data.get("channel", "R")
    if channel == "C":
        row["channel_C"] = 1
    elif channel == "R":
        row["channel_R"] = 1

    # ── Property type (one-hot, base = CO) ────────────────────────────
    prop_type = data.get("prop_type", "SF")
    if prop_type == "MH":
        row["prop_type_MH"] = 1
    elif prop_type == "PU":
        row["prop_type_PU"] = 1
    elif prop_type == "SF":
        row["prop_type_SF"] = 1
    elif prop_type == "CP":
        row["prop_type_CP"] = 1

    # ── Num units (one-hot, base = 1) ─────────────────────────────────
    num_units = int(data.get("num_units", 1))
    if num_units == 2:
        row["num_units_2"] = 1
    elif num_units == 3:
        row["num_units_3"] = 1
    elif num_units >= 4:
        row["num_units_4"] = 1

    # ── Return as ordered DataFrame row ──────────────────────────────
    return pd.DataFrame([row])[FEATURE_COLS]

# ── Top factors ──────────────────────────────────────────────────────────────
def get_top_factors(input_row):
    """Returns top 3 factors driving the prediction with direction."""
    coefs = MODEL.coef_[0]
    values = input_row.values[0]
    contributions = coefs * values
    feature_impact = list(zip(FEATURE_COLS, contributions))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    LABELS = {
        "credit_score":     "Credit Score",
        "dti":              "Debt-to-Income Ratio",
        "ltv":              "Loan-to-Value Ratio",
        "or_interest_rate": "Interest Rate",
        "num_borrowers":    "Number of Borrowers",
        "state_def_rate":   "State Default Rate",
        "first_time_buyer_Y": "First-Time Buyer",
        "mi_cancel":        "Mortgage Insurance",
        "metro_area":       "Metro Area",
        "super_conforming": "Super Conforming Loan",
    }

    top = []
    for feat, impact in feature_impact[:3]:
        top.append({
            "feature": LABELS.get(feat, feat.replace("_", " ").title()),
            "direction": "increases" if impact > 0 else "decreases",
            "impact": round(abs(impact), 4)
        })
    return top

# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    states = sorted(STATE_RATES.keys())
    return render_template("index.html", states=states)


@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint. Accepts JSON, returns prediction.

    Example request:
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
    """
    try:
        data = request.get_json(force=True)
        input_row = build_input_row(data)
        scaled = SCALER.transform(input_row)
        prob = float(MODEL.predict_proba(scaled)[0, 1])
        tier, label, explanation = assign_risk_tier(prob)
        top_factors = get_top_factors(input_row)

        return jsonify({
            "probability":   round(prob, 4),
            "probability_pct": f"{prob*100:.1f}%",
            "risk_tier":     tier,
            "risk_label":    label,
            "explanation":   explanation,
            "top_factors":   top_factors,
            "state_rate":    round(STATE_RATES.get(
                data.get("prop_state", "").upper(),
                NATIONAL_AVG_RATE), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": len(FEATURE_COLS)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
