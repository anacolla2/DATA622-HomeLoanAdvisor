# ============================================================================
# build_dataset.py
# Home Loan Advisor — Automated Training Pipeline
# ============================================================================
# Processes raw Freddie Mac files → cleaned dataset → trained model assets
#
# USAGE:
#   python build_dataset.py --data_dir data/ --sample 50000
#
# ARGUMENTS:
#   --data_dir   Directory containing raw Freddie Mac .txt files
#   --sample     Rows per file (default: None = use all rows)
#   --output_dir Directory to save model assets (default: model_assets/)
#
# EXPECTED FILE NAMING:
#   Origination:  historical_data_YYYYQN.txt
#   Performance:  historical_data_time_YYYYQN.txt
# ============================================================================

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# ── Cleaning functions ───────────────────────────────────────────────────────

def clean_origination(filepath, nrows=None):
    """
    Cleans a Freddie Mac Single Family Loan Level origination file.
    Expected: pipe-delimited .txt, no header.
    e.g. historical_data_2018Q1.txt
    """
    or_df = pd.read_csv(
        filepath, sep="|", header=None, nrows=nrows,
        na_values=["", "9999", "999", "9", "   ", "X"]
    )

    or_df.columns = [
        "credit_score", "first_payment_date", "first_time_buyer",
        "maturity_date", "msa", "mi_percent", "num_units", "occupancy",
        "cltv", "dti", "upb", "ltv", "or_interest_rate", "channel",
        "ppm", "amortization", "prop_state", "prop_type", "zipcode",
        "loan_id", "loan_purpose", "or_loan_term", "num_borrowers",
        "seller", "servicer", "super_conforming", "pre_harp_loan_id",
        "program_indicator", "harp_indicator", "prop_valuation",
        "interest_only", "mi_cancel"
    ]

    # remove HARP loans
    or_df = or_df[or_df["harp_indicator"].isna()].copy()

    # drop irrelevant columns
    drop = [
        "pre_harp_loan_id", "amortization", "harp_indicator",
        "program_indicator", "seller", "servicer", "first_payment_date",
        "maturity_date", "zipcode", "cltv"
    ]
    or_df = or_df.drop(columns=drop)
    or_df = or_df.set_index("loan_id")

    # msa → metro_area binary
    or_df["metro_area"] = or_df["msa"].notna().astype(int)
    or_df = or_df.drop(columns="msa")

    # super conforming binary
    or_df["super_conforming"] = (or_df["super_conforming"] == "Y").astype(int)

    # one-hot encode categoricals
    categorical_cols = [
        "first_time_buyer", "loan_purpose", "prop_type",
        "occupancy", "channel", "num_units"
    ]
    or_df = pd.get_dummies(
        or_df, columns=categorical_cols, drop_first=True, dtype=int
    )

    # numeric imputation
    for col in ["credit_score", "dti", "ltv", "mi_percent", "prop_valuation", "upb"]:
        if col in or_df.columns:
            or_df[col] = or_df[col].fillna(or_df[col].median())

    # flag encoding
    for col in ["interest_only", "ppm", "mi_cancel"]:
        if col in or_df.columns:
            or_df[col] = (or_df[col] == "Y").astype(int)

    return or_df


def clean_performance(filepath, nrows=None):
    """
    Cleans a Freddie Mac Single Family Loan Level performance file.
    Expected: pipe-delimited .txt, no header.
    e.g. historical_data_time_2018Q1.txt
    """
    perf_df = pd.read_csv(
        filepath, sep="|", header=None, nrows=nrows,
        na_values=["", "9999", "999", "9", "   ", "X"]
    )

    perf_df.columns = [
        "loan_id", "report_period", "current_upb", "delinquency_status",
        "loan_age", "remaining_months", "defect_settlement_date",
        "modification_flag", "zero_balance_code", "zero_balance_effective_date",
        "current_rate", "current_deferred_upb", "ddlpi", "mi_recoveries",
        "net_sales_proceeds", "non_mi_recoveries", "expenses", "legal_costs",
        "maintenance_costs", "taxes_insurance", "misc_expenses", "actual_loss",
        "modification_cost", "step_modification_flag", "deferred_payment_plan",
        "eltv", "zero_balance_removal_upb", "delinquent_accrued_interest",
        "disaster_delinquency", "borrower_assistance", "current_month_mod_cost",
        "interest_bearing_upb"
    ]

    keeping = [
        "loan_id", "report_period", "current_upb", "delinquency_status",
        "loan_age", "remaining_months", "zero_balance_code", "current_rate",
        "modification_flag", "disaster_delinquency", "borrower_assistance"
    ]
    perf_df = perf_df[keeping].set_index("loan_id")
    perf_df["delinquency_status"] = pd.to_numeric(
        perf_df["delinquency_status"], errors="coerce"
    )

    return perf_df


# ── Main pipeline ────────────────────────────────────────────────────────────

def build_dataset(data_dir, sample=None, output_dir="model_assets"):
    os.makedirs(output_dir, exist_ok=True)

    # find files
    all_files = os.listdir(data_dir)
    orig_files = sorted([
        f for f in all_files
        if f.startswith("historical_data_2") and "time" not in f and f.endswith(".txt")
    ])
    perf_files = sorted([
        f for f in all_files
        if f.startswith("historical_data_time") and f.endswith(".txt")
    ])

    print(f"Found {len(orig_files)} origination files")
    print(f"Found {len(perf_files)} performance files")

    if not orig_files or not perf_files:
        raise FileNotFoundError(
            "No Freddie Mac files found. "
            "Expected: historical_data_YYYYQN.txt and historical_data_time_YYYYQN.txt"
        )

    # ── Stack origination ──────────────────────────────────────────────
    print("\nLoading origination files...")
    or_frames = []
    for f in orig_files:
        print(f"  {f}")
        try:
            df = clean_origination(os.path.join(data_dir, f), nrows=sample)
            or_frames.append(df)
        except Exception as e:
            print(f"  WARNING: skipped {f} — {e}")

    or_all = pd.concat(or_frames, ignore_index=False)
    or_all = or_all[~or_all.index.duplicated(keep="first")]
    print(f"Origination shape: {or_all.shape}")

    # ── Stack performance ──────────────────────────────────────────────
    print("\nLoading performance files...")
    perf_frames = []
    for f in perf_files:
        print(f"  {f}")
        try:
            df = clean_performance(os.path.join(data_dir, f), nrows=sample)
            perf_frames.append(df)
        except Exception as e:
            print(f"  WARNING: skipped {f} — {e}")

    perf_all = pd.concat(perf_frames, ignore_index=False)
    print(f"Performance shape: {perf_all.shape}")

    # ── Build target ───────────────────────────────────────────────────
    print("\nBuilding target variable...")
    target = (
        perf_all.groupby("loan_id")["delinquency_status"]
        .max().ge(3).astype(int)
        .reset_index()
        .rename(columns={"delinquency_status": "any_90_dpd"})
    )

    zero_bal = (
        perf_all.groupby("loan_id")["zero_balance_code"]
        .max().reset_index()
        .rename(columns={"zero_balance_code": "termin_code"})
    )

    target = target.merge(zero_bal, on="loan_id", how="left")
    target["termin_code"] = target["termin_code"].fillna(0)

    # ── Final merge ────────────────────────────────────────────────────
    print("Merging origination + target...")
    final = or_all.merge(target, on="loan_id", how="inner")
    final = final.set_index("loan_id")
    print(f"Final shape: {final.shape}")
    print(f"Default rate: {final['any_90_dpd'].mean():.4%}")

    # ── State default rate ─────────────────────────────────────────────
    print("Computing state default rates...")
    state_def_rate = final.groupby("prop_state")["any_90_dpd"].mean()
    state_def_rate.to_csv(os.path.join(output_dir, "state_def_rate.csv"), header=True)

    final["state_def_rate"] = final["prop_state"].map(state_def_rate)
    final = final.drop(columns=["prop_state", "termin_code"], errors="ignore")

    # ── Drop weak features ─────────────────────────────────────────────
    drop_weak = [
        "prop_valuation", "or_loan_term", "interest_only",
        "ppm", "upb", "loan_purpose_N", "loan_purpose_P", "mi_percent"
    ]
    final = final.drop(columns=[c for c in drop_weak if c in final.columns])

    # ── Train ──────────────────────────────────────────────────────────
    print("\nTraining model...")
    X = final.drop(columns="any_90_dpd")
    y = final["any_90_dpd"]

    json.dump(list(X.columns), open(
        os.path.join(output_dir, "feature_columns.json"), "w"
    ))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nAUC-ROC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test_scaled)))

    # ── Save ───────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved to {output_dir}/:")
    print(os.listdir(output_dir))
    return final


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Home Loan Advisor dataset and model")
    parser.add_argument("--data_dir",   default="data/",          help="Raw data directory")
    parser.add_argument("--sample",     default=None, type=int,   help="Rows per file (None = all)")
    parser.add_argument("--output_dir", default="model_assets/",  help="Output directory")
    args = parser.parse_args()

    build_dataset(
        data_dir=args.data_dir,
        sample=args.sample,
        output_dir=args.output_dir
    )
