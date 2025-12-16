# -*- coding: utf-8 -*-
"""data_processing.py
Cleaned and consolidated version for feature engineering and target creation.
"""

import os
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# -----------------------------
# Feature Engineering Functions
# -----------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate features per customer and extract time-based features."""
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Aggregate per customer
    agg_df = df.groupby("CustomerId").agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count"),
        amount_std=("Amount", "std"),
        last_transaction=("TransactionStartTime", "max")
    ).reset_index()

    # Extract time features from last transaction
    agg_df["transaction_hour"] = agg_df["last_transaction"].dt.hour
    agg_df["transaction_day"] = agg_df["last_transaction"].dt.day
    agg_df["transaction_month"] = agg_df["last_transaction"].dt.month
    agg_df["transaction_year"] = agg_df["last_transaction"].dt.year

    agg_df.drop(columns=["last_transaction"], inplace=True)

    return agg_df


# -----------------------------
# RFM Target Creation Functions
# -----------------------------

def create_rfm_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a proxy target (is_high_risk) using RFM and KMeans clustering."""
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    snapshot_date = df["TransactionStartTime"].max() + timedelta(days=1)

    # Compute RFM features
    rfm = df.groupby("CustomerId").agg(
        recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Amount", "sum")
    ).reset_index()

    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    # Cluster customers
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster: lowest frequency and monetary
    cluster_summary = rfm.groupby("cluster")[["frequency", "monetary"]].mean()
    high_risk_cluster = cluster_summary.sort_values(by=["frequency", "monetary"]).index[0]

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]


# -----------------------------
# Utility Functions
# -----------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw transaction CSV."""
    return pd.read_csv(path)


def save_processed_data(df: pd.DataFrame, path: str):
    """Save processed dataset to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Processed data saved to {path}")


# -----------------------------
# Main Script
# -----------------------------

def main():
    raw_path = "data/raw/data.csv"  # Update if your raw CSV path differs
    processed_path = "data/processed/processed_data.csv"

    # Load raw data
    df_raw = load_raw_data(raw_path)

    # Build features
    df_features = build_features(df_raw)

    # Create RFM-based target
    target_df = create_rfm_target(df_raw)

    # Merge features and target
    df_final = df_features.merge(target_df, on="CustomerId", how="left")

    # Save final processed data
    save_processed_data(df_final, processed_path)


if __name__ == "__main__":
    main()
