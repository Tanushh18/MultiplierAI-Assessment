"""
Part 3 — Backend: FastAPI server
Serves analysis CSVs as JSON APIs.
Run with: uvicorn main:app --reload --port 8000
"""

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Data Dashboard API", version="1.0.0")

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config: data directory (relative to where uvicorn is run) ─────────────────
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data", "processed"))


def read_csv_safe(filename: str) -> list[dict]:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Data file '{filename}' not found. Run the pipeline first.",
        )
    try:
        df = pd.read_csv(path)
        # Replace NaN with None so JSON serialises cleanly
        df = df.where(pd.notnull(df), None)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "OK", "message": "Data Dashboard API is running"}


# ─── Revenue ──────────────────────────────────────────────────────────────────
@app.get("/api/revenue")
def get_revenue():
    data = read_csv_safe("monthly_revenue.csv")
    return JSONResponse(content={"success": True, "count": len(data), "data": data})


# ─── Top Customers ────────────────────────────────────────────────────────────
@app.get("/api/top-customers")
def get_top_customers():
    data = read_csv_safe("top_customers.csv")
    return JSONResponse(content={"success": True, "count": len(data), "data": data})


# ─── Categories ───────────────────────────────────────────────────────────────
@app.get("/api/categories")
def get_categories():
    data = read_csv_safe("category_performance.csv")
    return JSONResponse(content={"success": True, "count": len(data), "data": data})


# ─── Regions ──────────────────────────────────────────────────────────────────
@app.get("/api/regions")
def get_regions():
    data = read_csv_safe("regional_analysis.csv")
    return JSONResponse(content={"success": True, "count": len(data), "data": data})
