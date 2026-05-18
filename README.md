# BetaCalc — Portfolio Weighted Beta Engine

**Live OLS Beta calculation for Indian portfolios**  
Stocks (NSE/BSE) · ETFs · Mutual Funds · NIFTY 50 Benchmark

---

## 🚀 Deploy on Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub

1. Create a **new public (or private) GitHub repository** (e.g. `betacalc`)
2. Upload all files keeping this exact structure:

```
betacalc/
├── app.py
├── beta_engine.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

You can drag-and-drop all files directly into the GitHub web UI.

---

### Step 2 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"**
3. Fill in:
   - **Repository**: `your-username/betacalc`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **"Deploy!"**

That's it — Streamlit will install dependencies from `requirements.txt` automatically.  
Your app will be live at:  
`https://your-username-betacalc-app-xxxxxx.streamlit.app`

---

## 📋 CSV Format

```csv
Ticker,Weight,AssetType
RELIANCE.NS,15,Stock
TCS.NS,12,Stock
HDFC Top 100 Fund,10,MutualFund
GOLDBEES.NS,5,ETF
```

- **Ticker**: NSE ticker (`.NS`) or Mutual Fund name as listed on AMFI
- **Weight**: Any numeric weight (%, ₹ value, units — will be normalized to 100%)
- **AssetType**: `Stock` / `ETF` / `MutualFund` (optional — auto-detected)

Also accepts broker exports from **Zerodha / Groww / Kite** with columns like  
`Instrument`, `Cur. val`, `Symbol`, etc.

---

## ⚙️ Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📡 Data Sources

| Source | Used For |
|---|---|
| `yfinance` | NSE/BSE stocks and ETFs (adjusted daily closes) |
| `mfapi.in` | Mutual fund NAV history via AMFI scheme codes |
| `AMFI NAVAll` | Fuzzy-match fund names to scheme codes |

Benchmark: **NIFTY 50** (`^NSEI`)

---

> ⚠️ For informational purposes only. Not financial advice.  
> Beta is a historical measure and does not predict future risk.
