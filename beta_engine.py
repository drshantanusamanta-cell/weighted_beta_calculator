"""
beta_engine.py
==============
Core weighted beta calculation engine for Indian portfolios.

Data Sources:
  - yfinance       : NSE/BSE listed stocks and ETFs (suffix .NS / .BO)
  - mfapi.in       : Mutual fund NAV history (free AMFI-backed API)
  - AMFI name list : Fuzzy-match mutual fund names → scheme codes

Beta Calculation:
  - OLS regression of daily log returns vs benchmark (^NSEI = NIFTY 50)
  - Default lookback: 3 years of daily data
  - Minimum data requirement: 120 trading days
"""

import re
import time
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BENCHMARK_TICKER = "^NSEI"          # NIFTY 50
BENCHMARK_NAME   = "NIFTY 50"
LOOKBACK_YEARS   = 3
MIN_DATA_POINTS  = 120              # ~6 months of trading days
MFAPI_BASE       = "https://api.mfapi.in/mf"
AMFI_NAV_URL     = "https://www.amfiindia.com/spages/NAVAll.txt"
REQUEST_TIMEOUT  = 20
REQUEST_HEADERS  = {"User-Agent": "Mozilla/5.0 BetaCalc/1.0"}


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class HoldingResult:
    ticker:          str
    asset_type:      str
    raw_weight:      float
    norm_weight:     float           # normalized to sum=100
    beta:            Optional[float]
    r_squared:       Optional[float]
    data_points:     Optional[int]
    period_used:     Optional[str]
    data_source:     str             # "yfinance" | "mfapi" | "unavailable"
    benchmark:       str
    status:          str             # "ok" | "warn" | "error"
    note:            str
    scheme_code:     Optional[str]   = None
    scheme_name:     Optional[str]   = None

@dataclass
class PortfolioResult:
    holdings:          list[HoldingResult]
    weighted_beta:     Optional[float]      # over available holdings only
    coverage_pct:      float                # % of portfolio weight with valid beta
    total_holdings:    int
    available:         int
    unavailable:       int
    benchmark:         str
    calculation_date:  str
    warnings:          list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Benchmark loader
# ─────────────────────────────────────────────
_benchmark_cache: Optional[pd.Series] = None

def _load_benchmark(years: int = LOOKBACK_YEARS) -> pd.Series:
    global _benchmark_cache
    if _benchmark_cache is not None:
        return _benchmark_cache
    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 30)
    df = yf.download(BENCHMARK_TICKER, start=start, end=end,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("Could not fetch NIFTY 50 benchmark data.")
    closes = df["Close"].squeeze().dropna()
    _benchmark_cache = np.log(closes / closes.shift(1)).dropna()
    return _benchmark_cache


# ─────────────────────────────────────────────
# OLS Beta helper
# ─────────────────────────────────────────────
def _compute_beta(asset_returns: pd.Series,
                  bench_returns: pd.Series) -> tuple[float, float, int]:
    """Returns (beta, r_squared, n_points)."""
    common = asset_returns.index.intersection(bench_returns.index)
    a = asset_returns.loc[common].dropna()
    b = bench_returns.loc[common].dropna()
    common2 = a.index.intersection(b.index)
    a, b = a.loc[common2], b.loc[common2]
    n = len(a)
    if n < MIN_DATA_POINTS:
        raise ValueError(f"Insufficient data: {n} points (need {MIN_DATA_POINTS})")
    slope, _, r, _, _ = stats.linregress(b.values, a.values)
    return round(float(slope), 4), round(float(r ** 2), 4), n


# ─────────────────────────────────────────────
# Stock / ETF beta via yfinance
# ─────────────────────────────────────────────
def _beta_from_yfinance(ticker: str,
                         bench: pd.Series,
                         years: int = LOOKBACK_YEARS) -> HoldingResult:
    """Fetch OHLCV from yfinance, compute OLS beta vs NIFTY 50."""
    # Ensure .NS suffix for bare NSE tickers
    yf_ticker = ticker if ("." in ticker or "^" in ticker) else ticker + ".NS"
    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 30)

    try:
        df = yf.download(yf_ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty:
            # Try .BO (BSE) as fallback
            yf_ticker_bo = ticker.replace(".NS", "") + ".BO"
            df = yf.download(yf_ticker_bo, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                return None, "Not found on yfinance (tried .NS and .BO)"
            yf_ticker = yf_ticker_bo

        closes  = df["Close"].squeeze().dropna()
        returns = np.log(closes / closes.shift(1)).dropna()
        beta, r2, n = _compute_beta(returns, bench)
        period_str = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        return (beta, r2, n, period_str, yf_ticker), None

    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"yfinance error: {e}"


# ─────────────────────────────────────────────
# Mutual Fund beta via mfapi.in
# ─────────────────────────────────────────────
_amfi_list_cache: Optional[pd.DataFrame] = None

def _load_amfi_list() -> pd.DataFrame:
    """Fetch AMFI scheme list once and cache."""
    global _amfi_list_cache
    if _amfi_list_cache is not None:
        return _amfi_list_cache
    try:
        resp = requests.get(AMFI_NAV_URL, timeout=REQUEST_TIMEOUT,
                            headers=REQUEST_HEADERS)
        resp.raise_for_status()
        rows = []
        for line in resp.text.splitlines():
            parts = line.split(";")
            if len(parts) >= 4 and parts[0].strip().isdigit():
                rows.append({
                    "scheme_code": parts[0].strip(),
                    "scheme_name": parts[3].strip(),
                })
        _amfi_list_cache = pd.DataFrame(rows)
        return _amfi_list_cache
    except Exception as e:
        raise RuntimeError(f"Could not load AMFI scheme list: {e}")


def _fuzzy_match_fund(name: str, amfi_df: pd.DataFrame,
                       threshold: float = 0.45) -> Optional[tuple[str, str]]:
    """
    Simple token-overlap fuzzy match.
    Returns (scheme_code, scheme_name) or None.
    """
    name_tokens = set(re.sub(r"[^a-z0-9 ]", "", name.lower()).split())
    best_score, best_row = 0.0, None
    for _, row in amfi_df.iterrows():
        row_tokens = set(re.sub(r"[^a-z0-9 ]", "", row["scheme_name"].lower()).split())
        if not row_tokens:
            continue
        overlap = len(name_tokens & row_tokens) / max(len(name_tokens | row_tokens), 1)
        if overlap > best_score:
            best_score, best_row = overlap, row
    if best_score >= threshold and best_row is not None:
        return best_row["scheme_code"], best_row["scheme_name"]
    return None


def _fetch_mf_nav(scheme_code: str) -> Optional[pd.Series]:
    """Fetch NAV history from mfapi.in → daily Series."""
    try:
        url  = f"{MFAPI_BASE}/{scheme_code}"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna().set_index("date").sort_index()
        return df["nav"]
    except Exception:
        return None


def _beta_from_mfapi(fund_name: str,
                      bench: pd.Series,
                      years: int = LOOKBACK_YEARS) -> tuple:
    """Resolve fund name → scheme code → NAV history → OLS beta."""
    try:
        amfi_df = _load_amfi_list()
    except RuntimeError as e:
        return None, str(e), None, None

    match = _fuzzy_match_fund(fund_name, amfi_df)
    if match is None:
        return None, f"No AMFI match found for '{fund_name}' (fuzzy threshold not met)", None, None

    scheme_code, scheme_name = match
    nav = _fetch_mf_nav(scheme_code)
    if nav is None or nav.empty:
        return None, f"NAV data unavailable from mfapi (scheme {scheme_code})", scheme_code, scheme_name

    # Resample to business-day frequency, forward-fill (MF NAVs are daily but weekends absent)
    cutoff = datetime.today() - timedelta(days=years * 365 + 30)
    nav    = nav[nav.index >= cutoff]
    nav    = nav.resample("B").last().ffill()
    returns = np.log(nav / nav.shift(1)).dropna()

    try:
        beta, r2, n = _compute_beta(returns, bench)
    except ValueError as e:
        return None, str(e), scheme_code, scheme_name

    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    period_str = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    return (beta, r2, n, period_str), None, scheme_code, scheme_name


# ─────────────────────────────────────────────
# Asset type detection
# ─────────────────────────────────────────────
ETF_KEYWORDS = {"etf", "bees", "ietf", "liquidbees", "goldbees", "juniorbees",
                "niftybees", "bankbees", "setfnif50", "utiniftetf"}
MF_KEYWORDS  = {"fund", "scheme", "direct", "regular", "growth", "dividend",
                "idcw", "flexi", "cap", "equity", "debt", "liquid", "hybrid",
                "balanced", "elss", "tax"}

def _detect_asset_type(ticker: str, declared_type: str) -> str:
    """Return 'Stock', 'ETF', or 'MutualFund'."""
    dt = (declared_type or "").strip().lower()
    if "mutual" in dt or "mf" in dt or "fund" in dt:
        return "MutualFund"
    if "etf" in dt:
        return "ETF"
    if "stock" in dt or "equity" in dt or "share" in dt:
        return "Stock"

    # Heuristic on ticker string
    tl = ticker.lower()
    for kw in ETF_KEYWORDS:
        if kw in tl:
            return "ETF"
    # If ticker has spaces or common MF words → treat as MF
    if " " in ticker:
        for kw in MF_KEYWORDS:
            if kw in tl:
                return "MutualFund"
    return "Stock"


# ─────────────────────────────────────────────
# Main portfolio calculator
# ─────────────────────────────────────────────
def calculate_portfolio_beta(
    portfolio_df: pd.DataFrame,
    lookback_years: int = LOOKBACK_YEARS,
    progress_callback=None,          # fn(current, total, message)
) -> PortfolioResult:
    """
    portfolio_df must have columns: Ticker, Weight, AssetType (optional)
    Returns a PortfolioResult.
    """
    # ── Normalize columns ──────────────────────────────────────────────
    # Supports: standard format AND broker exports (Zerodha/Kite, Groww, etc.)
    col_map = {}
    for col in portfolio_df.columns:
        cl = col.strip().lower().replace(".", "").replace(" ", "")
        # Ticker / instrument name
        if any(k in cl for k in ["ticker", "symbol", "scrip", "instrument"]) or cl == "name":
            col_map["ticker"] = col
        # Weight: prefer current value columns over invested, then weight/allocation
        elif any(k in cl for k in ["curval", "currentval", "mktval", "marketval", "presentval"]):
            col_map["weight"] = col   # current market value → best proxy for weight
        elif any(k in cl for k in ["invested", "investment"]) and "weight" not in col_map:
            col_map["weight"] = col   # fallback to invested amount
        elif any(k in cl for k in ["weight", "alloc", "percent"]) and "weight" not in col_map:
            col_map["weight"] = col
        # Asset type
        elif any(k in cl for k in ["type", "assetclass", "assettype"]):
            col_map["type"] = col

    if "ticker" not in col_map or "weight" not in col_map:
        raise ValueError(
            f"Could not detect Ticker and Weight columns.\n"
            f"Columns found: {list(portfolio_df.columns)}\n"
            f"Expected columns like: Instrument/Ticker/Symbol  AND  "
            f"Cur. val / Weight / Invested / Allocation."
        )

    df = portfolio_df.rename(columns={
        col_map["ticker"]: "ticker",
        col_map["weight"]: "weight",
    })
    if "type" in col_map:
        df = df.rename(columns={col_map["type"]: "asset_type"})
    else:
        df["asset_type"] = "Unknown"

    df["ticker"]     = df["ticker"].astype(str).str.strip()
    df["weight"]     = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    df["asset_type"] = df["asset_type"].astype(str).str.strip()
    df = df[df["weight"] > 0].reset_index(drop=True)

    total_w = df["weight"].sum()
    df["norm_weight"] = df["weight"] / total_w * 100

    # ── Load benchmark once ───────────────────────────────────────────
    if progress_callback:
        progress_callback(0, len(df), "Loading NIFTY 50 benchmark data…")
    try:
        bench = _load_benchmark(lookback_years)
    except RuntimeError as e:
        raise RuntimeError(f"Benchmark load failed: {e}")

    results   = []
    warns_all = []

    for i, row in df.iterrows():
        ticker     = row["ticker"]
        raw_w      = row["weight"]
        norm_w     = row["norm_weight"]
        asset_type = _detect_asset_type(ticker, row["asset_type"])

        if progress_callback:
            progress_callback(i + 1, len(df), f"Processing {ticker}…")

        # ── Stock or ETF ───────────────────────────────────────────────
        if asset_type in ("Stock", "ETF"):
            result, err = _beta_from_yfinance(ticker, bench, lookback_years)
            if err:
                results.append(HoldingResult(
                    ticker=ticker, asset_type=asset_type,
                    raw_weight=raw_w, norm_weight=norm_w,
                    beta=None, r_squared=None, data_points=None,
                    period_used=None, data_source="unavailable",
                    benchmark=BENCHMARK_NAME, status="error", note=err,
                ))
                warns_all.append(f"{ticker}: {err}")
            else:
                beta, r2, n, period, used_ticker = result
                note = ""
                if used_ticker != ticker and used_ticker != ticker + ".NS":
                    note = f"Resolved to {used_ticker}"
                results.append(HoldingResult(
                    ticker=ticker, asset_type=asset_type,
                    raw_weight=raw_w, norm_weight=norm_w,
                    beta=beta, r_squared=r2, data_points=n,
                    period_used=period, data_source="yfinance",
                    benchmark=BENCHMARK_NAME, status="ok", note=note,
                ))

        # ── Mutual Fund ────────────────────────────────────────────────
        elif asset_type == "MutualFund":
            result, err, sc, sn = _beta_from_mfapi(ticker, bench, lookback_years)
            if err:
                results.append(HoldingResult(
                    ticker=ticker, asset_type=asset_type,
                    raw_weight=raw_w, norm_weight=norm_w,
                    beta=None, r_squared=None, data_points=None,
                    period_used=None, data_source="unavailable",
                    benchmark=BENCHMARK_NAME, status="error",
                    note=err, scheme_code=sc, scheme_name=sn,
                ))
                warns_all.append(f"{ticker}: {err}")
            else:
                beta, r2, n, period = result
                note = f"AMFI scheme: {sn} (code {sc})" if sn else ""
                results.append(HoldingResult(
                    ticker=ticker, asset_type=asset_type,
                    raw_weight=raw_w, norm_weight=norm_w,
                    beta=beta, r_squared=r2, data_points=n,
                    period_used=period, data_source="mfapi",
                    benchmark=BENCHMARK_NAME, status="ok",
                    note=note, scheme_code=sc, scheme_name=sn,
                ))

        else:
            results.append(HoldingResult(
                ticker=ticker, asset_type=asset_type,
                raw_weight=raw_w, norm_weight=norm_w,
                beta=None, r_squared=None, data_points=None,
                period_used=None, data_source="unavailable",
                benchmark=BENCHMARK_NAME, status="error",
                note="Unknown asset type — cannot determine data source.",
            ))

    # ── Weighted beta ─────────────────────────────────────────────────
    available = [r for r in results if r.beta is not None]
    coverage  = sum(r.norm_weight for r in available)
    if available:
        w_beta = sum(r.beta * r.norm_weight for r in available) / coverage
        w_beta = round(w_beta, 4)
    else:
        w_beta = None

    return PortfolioResult(
        holdings=results,
        weighted_beta=w_beta,
        coverage_pct=round(coverage, 2),
        total_holdings=len(results),
        available=len(available),
        unavailable=len(results) - len(available),
        benchmark=BENCHMARK_NAME,
        calculation_date=datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        warnings=warns_all,
    )


def results_to_dataframe(pr: PortfolioResult) -> pd.DataFrame:
    """Convert PortfolioResult to a flat DataFrame for export."""
    rows = []
    for h in pr.holdings:
        rows.append({
            "Ticker / Fund":   h.ticker,
            "Asset Type":      h.asset_type,
            "Raw Weight":      h.raw_weight,
            "Norm Weight (%)": round(h.norm_weight, 2),
            "Beta":            h.beta,
            "R-Squared":       h.r_squared,
            "Data Points":     h.data_points,
            "Period Used":     h.period_used,
            "Data Source":     h.data_source,
            "Benchmark":       h.benchmark,
            "Status":          h.status,
            "Notes":           h.note,
            "Scheme Code":     h.scheme_code,
            "Matched Fund":    h.scheme_name,
        })
    return pd.DataFrame(rows)
