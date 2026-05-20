import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


ORIGINATION_COLS= [
    "credit_score", "first_payment_date", "first_time_buyer", "maturity_date", "msa", "mi_percent", "num_units", "occupancy", "cltv", "dti", "upb",
    "ltv", "or_interest_rate", "channel", "ppm", "amortization", "prop_state", "prop_type", "zipcode", "loan_id", "loan_purpose", "or_loan_term",
    "num_borrowers", "seller", "servicer", "super_conforming", "pre_harp_loan_id", "program_indicator", "harp_indicator", "prop_valuation",
    "interest_only", "mi_cancel"
]

PERFORMANCE_COLS= [
    "loan_id", "report_period", "current_upb", "delinquency_status", "loan_age", "remaining_months", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date", "current_rate", "current_deferred_upb", "ddlpi", "mi_recoveries", "net_sales_proceeds",
    "non_mi_recoveries", "expenses", "legal_costs", "maintenance_costs", "taxes_insurance", "misc_expenses", "actual_loss", "modification_cost",
    "step_modification_flag", "deferred_payment_plan", "eltv", "zero_balance_removal_upb", "delinquent_accrued_interest", "disaster_delinquency",
    "borrower_assistance", "current_month_mod_cost", "interest_bearing_upb"
]

PERFORMANCE_KEEP= [
    "loan_id", "report_period", "current_upb", "delinquency_status", "loan_age", "remaining_months", "zero_balance_code", "current_rate",
    "modification_flag", "disaster_delinquency", "borrower_assistance", "eltv", "deferred_payment_plan"
]

OR_DROP= [
    "pre_harp_loan_id", "amortization", "harp_indicator", "program_indicator", "seller", "servicer", "first_payment_date", "maturity_date", "zipcode", "cltv"
]

OR_ZERO_VARIANCE= ["ppm", "super_conforming", "interest_only"]

OR_CATEGORICAL= ["first_time_buyer", "loan_purpose", "prop_type", "occupancy", "channel", "num_units"]

OR_NUMERIC_IMPUTE= ["credit_score", "dti", "ltv", "mi_percent", "prop_valuation", "upb"]


def clean_origination(path):
    path= os.path.expanduser(path)

    df= pd.read_csv(path, sep="|", header=None, na_values=["", "9999", "999", "9", "   ", "X"], low_memory=False)
    df.columns= ORIGINATION_COLS

    #-- date features--
    first_payment= pd.to_datetime(df["first_payment_date"], format="%Y%m", errors="coerce")
    maturity= pd.to_datetime(df["maturity_date"], format="%Y%m", errors="coerce")

    df["first_payment_year"]= first_payment.dt.year
    df["first_payment_month"]= first_payment.dt.month
    df["maturity_year"]= maturity.dt.year
    df["maturity_month"]= maturity.dt.month

    #-- remove HARP loans--
    df= df[df["harp_indicator"].isna()].copy()

    #-- drop irrelevant columns--
    df= df.drop(columns=OR_DROP)

    #-- msa to metro area binary--
    df["metro_area"]= df["msa"].notna().astype(int)
    df= df.drop(columns="msa")

    #-- one-hot encode--
    df= pd.get_dummies(df, columns=OR_CATEGORICAL, drop_first=True, dtype=int)

    #-- median imputation--
    for col in OR_NUMERIC_IMPUTE:
        if col in df.columns:
            df[col]= df[col].fillna(df[col].median())

    #-- drop zero-variance columns--
    existing_zero_var= [c for c in OR_ZERO_VARIANCE if c in df.columns]
    if existing_zero_var:
        df= df.drop(columns=existing_zero_var)

    return df


def clean_performance(path, chunksize=250000):
    path= os.path.expanduser(path)

    keep_idx= [PERFORMANCE_COLS.index(c) for c in PERFORMANCE_KEEP]

    target_parts= []
    zero_parts= []

    for chunk in pd.read_csv(path, sep="|", header=None, usecols=keep_idx, names=PERFORMANCE_KEEP, na_values=["", "9999", "999", "9", "   ", "X"], chunksize=chunksize, low_memory=False):
        #-- any_90_dpd target part--
        dq_num= pd.to_numeric(chunk["delinquency_status"].replace({"RA": 99}), errors="coerce")
        target_chunk= dq_num.groupby(chunk["loan_id"]).max().reset_index(name="max_dq")
        target_parts.append(target_chunk)

        #-- last observed zero_balance_code part--
        zero_chunk= chunk.dropna(subset=["zero_balance_code"]).copy()

        if not zero_chunk.empty:
            zero_chunk= zero_chunk.sort_values(["loan_id", "report_period"])
            zero_chunk= zero_chunk.groupby("loan_id")[["report_period", "zero_balance_code"]].last().reset_index()
            zero_parts.append(zero_chunk)

        print(f"Processed performance chunk: {chunk.shape}")

    #-- combine target chunks--
    target= pd.concat(target_parts, ignore_index=True)
    target= target.groupby("loan_id")["max_dq"].max().ge(3).astype(int).reset_index(name="any_90_dpd")

    #-- combine zero balance chunks--
    if zero_parts:
        zero_bal= pd.concat(zero_parts, ignore_index=True)
        zero_bal= zero_bal.sort_values(["loan_id", "report_period"])
        zero_bal= zero_bal.groupby("loan_id")["zero_balance_code"].last().reset_index()
        zero_bal= zero_bal.rename(columns={"zero_balance_code": "termin_code"})
    else:
        zero_bal= pd.DataFrame(columns=["loan_id", "termin_code"])

    target= target.merge(zero_bal, on="loan_id", how="left")
    target["termin_code"]= target["termin_code"].fillna(0)

    return target


def build_quarter(year, quarter, data_dir, out_dir):
    data_dir= os.path.expanduser(data_dir)
    out_dir= os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tag= f"{year}Q{quarter}"
    print(f"\nStarting {tag}...")

    or_path= os.path.join(data_dir, f"historical_data_{tag}.txt")
    perf_path= os.path.join(data_dir, f"historical_data_time_{tag}.txt")
    out_path= os.path.join(out_dir, f"fdf_{tag}.csv")

    if os.path.exists(out_path):
        print(f"Already exists, skipping {tag}: {out_path}")
        return out_path

    if not os.path.exists(or_path) or not os.path.exists(perf_path):
        print(f"skipping {tag}, file(s) not found")
        return None

    or_df= clean_origination(or_path)
    print(f"Origination cleaned for {tag}: {or_df.shape}")

    perf_df= clean_performance(perf_path)
    print(f"Performance cleaned for {tag}: {perf_df.shape}")

    merged= or_df.merge(perf_df, on="loan_id", how="inner")
    print(f"Merged {tag}: {merged.shape}")

    merged["quarter_tag"]= tag
    merged.to_csv(out_path, index=False)

    print(f"Saved {tag} to {out_path}")

    return out_path


def merge_all_quarters(out_dir, final_path, test_size=0.3, random_state=212):
    out_dir= os.path.expanduser(out_dir)
    final_path= os.path.expanduser(final_path)
    os.makedirs(final_path, exist_ok=True)

    quarter_files= sorted(glob.glob(os.path.join(out_dir, "fdf_*.csv")))
    if not quarter_files:
        raise FileNotFoundError(f"no fdf_*.csv files found in: {out_dir}")

    temp_dir= os.path.join(final_path, "temp_split")
    os.makedirs(temp_dir, exist_ok=True)

    state_parts= []
    global_sum= 0
    global_count= 0

    #-- first pass: split each quarter and calculate train-only state rates--
    for f in quarter_files:
        print(f"\nReading {f}")
        df= pd.read_csv(f, low_memory=False)
        df= df.fillna(0)

        print(f"Splitting {os.path.basename(f)}: {df.shape}")

        df["strata"]= df["quarter_tag"].astype(str) + "_" + df["any_90_dpd"].astype(str)

        train_q, test_q= train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["strata"])

        train_q= train_q.copy()
        test_q= test_q.copy()

        train_q= train_q.drop(columns="strata")
        test_q= test_q.drop(columns="strata")

        state_q= train_q.groupby("prop_state")["any_90_dpd"].agg(["sum", "count"]).reset_index()
        state_parts.append(state_q)

        global_sum += train_q["any_90_dpd"].sum()
        global_count += train_q["any_90_dpd"].count()

        train_path= os.path.join(temp_dir, f"train_{os.path.basename(f)}")
        test_path= os.path.join(temp_dir, f"test_{os.path.basename(f)}")

        train_q.to_csv(train_path, index=False)
        test_q.to_csv(test_path, index=False)

        print(f"Saved train part: {train_q.shape}")
        print(f"Saved test part: {test_q.shape}")

        del df, train_q, test_q

    #-- calculate state default rate from train only--
    state_counts= pd.concat(state_parts, ignore_index=True)
    state_def_rate= state_counts.groupby("prop_state")[["sum", "count"]].sum().reset_index()
    state_def_rate["state_def_rate"]= state_def_rate["sum"] / state_def_rate["count"]
    state_def_rate= state_def_rate[["prop_state", "state_def_rate"]]

    global_def_rate= global_sum / global_count
    state_lookup= dict(zip(state_def_rate["prop_state"], state_def_rate["state_def_rate"]))

    state_def_rate.to_csv(os.path.join(final_path, "state_def_rate.csv"), index=False)

    #-- final output paths--
    train_out= os.path.join(final_path, "train_final.csv")
    test_out= os.path.join(final_path, "test_final.csv")

    if os.path.exists(train_out):
        os.remove(train_out)

    if os.path.exists(test_out):
        os.remove(test_out)

    train_files= sorted(glob.glob(os.path.join(temp_dir, "train_fdf_*.csv")))
    test_files= sorted(glob.glob(os.path.join(temp_dir, "test_fdf_*.csv")))

    drop_cols= ["loan_id", "prop_state", "termin_code"]

    #-- second pass: add state_def_rate and append train files--
    print("\nWriting final train file...")
    for i, f in enumerate(train_files):
        df= pd.read_csv(f, low_memory=False)
        df["state_def_rate"]= df["prop_state"].map(state_lookup).fillna(global_def_rate)
        df= df.drop(columns=[c for c in drop_cols if c in df.columns])

        df.to_csv(train_out, mode="a", header=(i == 0), index=False)
        print(f"Appended {os.path.basename(f)}: {df.shape}")

        del df

    #-- second pass: add state_def_rate and append test files--
    print("\nWriting final test file...")
    for i, f in enumerate(test_files):
        df= pd.read_csv(f, low_memory=False)
        df["state_def_rate"]= df["prop_state"].map(state_lookup).fillna(global_def_rate)
        df= df.drop(columns=[c for c in drop_cols if c in df.columns])

        df.to_csv(test_out, mode="a", header=(i == 0), index=False)
        print(f"Appended {os.path.basename(f)}: {df.shape}")

        del df

    #-- check final files lightly--
    print("\nFinal files saved:")
    print(train_out)
    print(test_out)
    print(os.path.join(final_path, "state_def_rate.csv"))

    print("\nDone.")

    return train_out, test_out


if __name__ == "__main__":
    data_dir= "~/projects/data/homeloanadvisor"
    out_dir= "~/projects/data/homeloanadvisor/processed_quarters"
    final_path= "~/projects/data/homeloanadvisor/final_model_data"

    for year in range(2018, 2023):
        for quarter in range(1, 5):
            build_quarter(year, quarter, data_dir, out_dir)

    merge_all_quarters(out_dir, final_path)
