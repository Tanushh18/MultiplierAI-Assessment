"""
Part 1: Data Cleaning Script
Cleans customers.csv and orders.csv with full logging and reporting.
"""

import pandas as pd
import numpy as np
import re
import logging
import os
import argparse
from pathlib import Path

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cleaning.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Config / Argument Parsing ────────────────────────────────────────────────
def get_config():
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--clean-dir", default="data/clean", help="Path to output clean data directory")
    args = parser.parse_args()
    return args


# ─── Utility: safe CSV load ───────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded '{path}' — {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading '{path}': {e}")
        raise


# ─── Utility: cleaning report ─────────────────────────────────────────────────
def print_report(name: str, df_before: pd.DataFrame, df_after: pd.DataFrame, dupes_removed: int):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  CLEANING REPORT — {name}")
    print(sep)
    print(f"  Rows before cleaning    : {len(df_before)}")
    print(f"  Rows after cleaning     : {len(df_after)}")
    print(f"  Duplicates removed      : {dupes_removed}")
    print(f"\n  Null values BEFORE:")
    for col, cnt in df_before.isnull().sum().items():
        if cnt > 0:
            print(f"    {col:25s}: {cnt}")
    print(f"\n  Null values AFTER:")
    null_after = df_after.isnull().sum()
    has_nulls = False
    for col, cnt in null_after.items():
        if cnt > 0:
            print(f"    {col:25s}: {cnt}")
            has_nulls = True
    if not has_nulls:
        print("    (none)")
    print(sep + "\n")


# ─── Part 1A: Clean customers.csv ─────────────────────────────────────────────
def clean_customers(raw_path: str, clean_path: str):
    logger.info("─── Starting customers cleaning ───")
    df = load_csv(raw_path)
    df_before = df.copy()

    # 1. Remove duplicates by customer_id, keep latest signup_date
    #    First try parsing signup_date so we can sort
    df["signup_date"] = df["signup_date"].astype(str).str.strip()

    def parse_date_flexible(date_str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"):
            try:
                return pd.to_datetime(date_str, format=fmt)
            except Exception:
                pass
        try:
            return pd.to_datetime(date_str, dayfirst=True)
        except Exception:
            logger.warning(f"Could not parse date: '{date_str}' — setting to NaT")
            return pd.NaT

    df["signup_date_parsed"] = df["signup_date"].apply(parse_date_flexible)

    # Sort so latest signup_date is last, then keep last duplicate
    df_sorted = df.sort_values("signup_date_parsed", na_position="first")
    before_dedup = len(df_sorted)
    df_sorted = df_sorted.drop_duplicates(subset=["customer_id"], keep="last")
    dupes_removed = before_dedup - len(df_sorted)
    logger.info(f"Duplicates removed (customer_id): {dupes_removed}")

    df = df_sorted.copy()

    # 2. Lowercase emails
    df["email"] = df["email"].astype(str).str.strip().str.lower()

    # 3. Validate emails
    def is_valid_email(email: str) -> bool:
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))

    df["is_valid_email"] = df["email"].apply(is_valid_email)
    invalid_count = (~df["is_valid_email"]).sum()
    logger.info(f"Invalid emails found: {invalid_count}")

    # 4. Standardise signup_date → YYYY-MM-DD, nullify unparseable
    df["signup_date"] = df["signup_date_parsed"].dt.strftime("%Y-%m-%d")
    df.drop(columns=["signup_date_parsed"], inplace=True)

    # 5. Strip extra spaces from name and region
    df["name"] = df["name"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()

    # 6. Replace missing / empty / 'nan' region with "Unknown"
    df["region"] = df["region"].replace({"": "Unknown", "nan": "Unknown", "NaN": "Unknown"})
    df["region"] = df["region"].fillna("Unknown")

    # Save
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    logger.info(f"Saved cleaned customers → '{clean_path}'")

    print_report("customers.csv", df_before, df, dupes_removed)
    return df


# ─── Part 1B: Clean orders.csv ────────────────────────────────────────────────
def clean_orders(raw_path: str, clean_path: str):
    logger.info("─── Starting orders cleaning ───")
    df = load_csv(raw_path)
    df_before = df.copy()

    # 1. Parse order_date in multiple formats
    def parse_order_date(date_str):
        date_str = str(date_str).strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"):
            try:
                return pd.to_datetime(date_str, format=fmt)
            except Exception:
                pass
        try:
            return pd.to_datetime(date_str, dayfirst=True)
        except Exception:
            logger.warning(f"Could not parse order_date: '{date_str}'")
            return pd.NaT

    df["order_date"] = df["order_date"].apply(parse_order_date)

    # 2. Remove rows where BOTH order_id AND customer_id are missing
    mask_both_missing = df["order_id"].isna() & df["customer_id"].isna()
    rows_removed = mask_both_missing.sum()
    df = df[~mask_both_missing].copy()
    logger.info(f"Rows removed (both order_id & customer_id null): {rows_removed}")

    # 3. Fill missing amount using median grouped by product
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    product_medians = df.groupby("product")["amount"].median()
    logger.info("Product medians used for imputation:")
    for prod, med in product_medians.items():
        logger.info(f"  {prod}: {med}")

    def fill_amount(row):
        if pd.isna(row["amount"]):
            return product_medians.get(row["product"], np.nan)
        return row["amount"]

    df["amount"] = df.apply(fill_amount, axis=1)

    # 4. Normalise status values
    STATUS_MAP = {
        "done": "completed",
        "complete": "completed",
        "completed": "completed",
        "pending": "pending",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "cancel": "cancelled",
        "refunded": "refunded",
        "refund": "refunded",
    }

    def normalise_status(val):
        if pd.isna(val):
            return "pending"
        cleaned = str(val).strip().lower()
        return STATUS_MAP.get(cleaned, cleaned)

    df["status"] = df["status"].apply(normalise_status)
    logger.info(f"Status distribution after normalisation:\n{df['status'].value_counts().to_string()}")

    # 5. Create order_year_month column
    df["order_year_month"] = df["order_date"].dt.to_period("M").astype(str)

    dupes_removed = df_before.duplicated(subset=["order_id"]).sum()

    # Save
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    logger.info(f"Saved cleaned orders → '{clean_path}'")

    print_report("orders.csv", df_before, df, dupes_removed)
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    raw = cfg.raw_dir
    clean = cfg.clean_dir

    customers_df = clean_customers(
        raw_path=os.path.join(raw, "customers.csv"),
        clean_path=os.path.join(clean, "customers_clean.csv"),
    )

    orders_df = clean_orders(
        raw_path=os.path.join(raw, "orders.csv"),
        clean_path=os.path.join(clean, "orders_clean.csv"),
    )

    print("✅ Part 1 complete. Clean files saved to:", clean)
