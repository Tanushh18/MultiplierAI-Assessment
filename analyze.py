"""
Part 2: Data Merging & Analysis Script
Merges cleaned datasets and produces business insight CSVs.
"""

import pandas as pd
import numpy as np
import logging
import os
import argparse
from datetime import timedelta

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("analysis.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────
def get_config():
    parser = argparse.ArgumentParser(description="Data Analysis Pipeline")
    parser.add_argument("--clean-dir", default="data/clean", help="Path to clean data directory")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw data (products.csv)")
    parser.add_argument("--output-dir", default="data/analyzed", help="Output directory for analysis files")
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded '{path}' — {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading '{path}': {e}")
        raise


# ─── Step 1: Merge Datasets ───────────────────────────────────────────────────
def merge_datasets(orders_df, customers_df, products_df):
    logger.info("─── Merging datasets ───")

    # Merge orders ← customers on customer_id (left join to keep all orders)
    merged = pd.merge(
        orders_df,
        customers_df,
        on="customer_id",
        how="left",
        suffixes=("_order", "_customer"),
    )

    orders_no_customer = merged["name"].isna().sum()
    logger.info(f"Orders without a matching customer: {orders_no_customer}")
    print(f"\n  ⚠  Orders without matching customer : {orders_no_customer}")

    # Merge result ← products on product name (left join)
    merged = pd.merge(
        merged,
        products_df,
        on="product",
        how="left",
    )

    orders_no_product = merged["category"].isna().sum()
    logger.info(f"Orders without a matching product: {orders_no_product}")
    print(f"  ⚠  Orders without matching product  : {orders_no_product}\n")

    return merged


# ─── Step 2A: Monthly Revenue ─────────────────────────────────────────────────
def compute_monthly_revenue(merged_df, output_path):
    completed = merged_df[merged_df["status"] == "completed"].copy()
    monthly = (
        completed.groupby("order_year_month", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "total_revenue"})
        .sort_values("order_year_month")
    )
    monthly.to_csv(output_path, index=False)
    logger.info(f"Monthly revenue saved → '{output_path}' ({len(monthly)} rows)")
    return monthly


# ─── Step 2B: Top 10 Customers ────────────────────────────────────────────────
def compute_top_customers(merged_df, output_path):
    # Reference date = latest order date in dataset
    ref_date = pd.to_datetime(merged_df["order_date"]).max()
    cutoff = ref_date - timedelta(days=90)
    logger.info(f"Churn reference date: {ref_date.date()}, cutoff (90 days back): {cutoff.date()}")

    completed = merged_df[merged_df["status"] == "completed"].copy()
    completed["order_date_dt"] = pd.to_datetime(completed["order_date"])

    # Latest completed order date per customer
    latest_order = (
        completed.groupby("customer_id")["order_date_dt"]
        .max()
        .reset_index()
        .rename(columns={"order_date_dt": "latest_order_date"})
    )

    # Total spend per customer
    spend = (
        completed.groupby(["customer_id", "name", "region"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "total_spend"})
    )

    top = (
        spend.merge(latest_order, on="customer_id", how="left")
        .sort_values("total_spend", ascending=False)
        .head(10)
    )

    # Churn flag
    top["churned"] = top["latest_order_date"] < cutoff

    top.to_csv(output_path, index=False)
    logger.info(f"Top customers saved → '{output_path}'")
    return top


# ─── Step 2C: Category Performance ───────────────────────────────────────────
def compute_category_performance(merged_df, output_path):
    df = merged_df.dropna(subset=["category"]).copy()
    perf = (
        df.groupby("category")
        .agg(
            total_revenue=("amount", "sum"),
            average_order_value=("amount", "mean"),
            number_of_orders=("order_id", "count"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    perf["average_order_value"] = perf["average_order_value"].round(2)
    perf.to_csv(output_path, index=False)
    logger.info(f"Category performance saved → '{output_path}'")
    return perf


# ─── Step 2D: Regional Analysis ───────────────────────────────────────────────
def compute_regional_analysis(merged_df, customers_df, output_path):
    # Number of customers per region
    cust_region = (
        customers_df.groupby("region")["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "num_customers"})
    )

    # Order & revenue metrics per region
    order_region = (
        merged_df.groupby("region")
        .agg(
            num_orders=("order_id", "count"),
            total_revenue=("amount", "sum"),
        )
        .reset_index()
    )

    regional = pd.merge(cust_region, order_region, on="region", how="outer").fillna(0)
    regional["avg_revenue_per_customer"] = (
        regional["total_revenue"] / regional["num_customers"]
    ).round(2)

    regional.to_csv(output_path, index=False)
    logger.info(f"Regional analysis saved → '{output_path}'")
    return regional


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load clean data
    orders_df = load_csv(os.path.join(cfg.clean_dir, "orders_clean.csv"))
    customers_df = load_csv(os.path.join(cfg.clean_dir, "customers_clean.csv"))
    products_df = load_csv(os.path.join(cfg.raw_dir, "products.csv"))

    # Merge
    merged = merge_datasets(orders_df, customers_df, products_df)

    # Analysis
    compute_monthly_revenue(merged, os.path.join(cfg.output_dir, "monthly_revenue.csv"))
    compute_top_customers(merged, os.path.join(cfg.output_dir, "top_customers.csv"))
    compute_category_performance(merged, os.path.join(cfg.output_dir, "category_performance.csv"))
    compute_regional_analysis(merged, customers_df, os.path.join(cfg.output_dir, "regional_analysis.csv"))

    print("✅ Part 2 complete. Analysis files saved to:", cfg.output_dir)
