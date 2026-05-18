"""
app.py  —  BetaCalc Streamlit Dashboard
========================================
Run with:  streamlit run app.py

Requires: beta_engine.py in the same directory.
"""

import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from beta_engine import (
    calculate_portfolio_beta,
    results_to_dataframe,
    BENCHMARK_NAME,
    LOOKBACK_YEARS,
    MIN_DATA_POINTS,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BetaCalc — Portfolio Beta Engine",
    page_icon="β",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS  — clean light theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F8F9FC;
    color: #1A202C;
}

.stApp { background-color: #F8F9FC; }

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] * { color: #1A202C !important; }

[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
[data-testid="metric-container"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #718096 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #1A202C !important;
}

.stButton > button {
    background: #3B5BDB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 28px !important;
}
.stButton > button:hover { background: #2F4AC7 !important; }

[data-testid="stDownloadButton"] > button {
    background: #FFFFFF !important;
    color: #3B5BDB !important;
    border: 1.5px solid #3B5BDB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

.stProgress > div > div { background-color: #3B5BDB !important; }

.stAlert {
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}

[data-testid="stFileUploadDropzone"] {
    background: #FFFFFF !important;
    border: 2px dashed #CBD5E0 !important;
    border-radius: 12px !important;
}

[data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #E2E8F0 !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #718096 !important;
    border-radius: 7px !important;
    padding: 8px 16px !important;
}
[aria-selected="true"] {
    background: #3B5BDB !important;
    color: #FFFFFF !important;
    border-bottom-color: transparent !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #EEF2FF !important;
    color: #3B5BDB !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 12px !important;
}

hr { border-color: #E2E8F0 !important; margin: 1.5rem 0 !important; }

.stCaption { color: #A0AEC0 !important; font-size: 12px !important; }

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4 {
    color: #1A202C !important;
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def beta_color(beta):
    if beta is None: return "#A0AEC0"
    if beta < 0.7:   return "#38A169"
    if beta < 1.2:   return "#3B5BDB"
    if beta < 1.8:   return "#DD6B20"
    return "#E53E3E"

def beta_label(beta):
    if beta is None: return "N/A"
    if beta < 0.7:   return "🟢 Low Risk"
    if beta < 1.2:   return "🔵 Moderate"
    if beta < 1.8:   return "🟠 Aggressive"
    return "🔴 Very High"

def risk_profile(wb):
    if wb is None: return "Unknown",               "#A0AEC0"
    if wb < 0.6:   return "Conservative",          "#38A169"
    if wb < 0.9:   return "Moderate-Conservative", "#48BB78"
    if wb < 1.1:   return "Market-Neutral",        "#3B5BDB"
    if wb < 1.4:   return "Aggressive Growth",     "#DD6B20"
    return         "High Risk / Speculative",       "#E53E3E"

PLOT_BASE = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8F9FC",
    font=dict(family="Inter", color="#4A5568", size=12),
    margin=dict(t=24, b=48, l=48, r=24),
)

SAMPLE_CSV = """Ticker,Weight,AssetType
RELIANCE.NS,15,Stock
TCS.NS,12,Stock
HDFCBANK.NS,10,Stock
INFY.NS,8,Stock
WIPRO.NS,5,Stock
HDFC Top 100 Fund,10,MutualFund
Axis Bluechip Fund,8,MutualFund
MIRAEEMERGBLUE.NS,7,ETF
GOLDBEES.NS,5,ETF
LIQUIDBEES.NS,5,ETF
IRCTC.NS,15,Stock
"""


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:8px 0 4px">
  <div style="font-size:32px;font-weight:700;color:#3B5BDB;line-height:1">β</div>
  <div style="font-size:18px;font-weight:700;color:#1A202C;margin-top:4px">BetaCalc</div>
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;color:#A0AEC0;
              text-transform:uppercase;margin-top:3px">Portfolio Beta Engine</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    lookback = st.slider("Lookback period (years)", 1, 5, LOOKBACK_YEARS,
                         help="Historical window for OLS regression vs NIFTY 50")
    st.caption(f"Benchmark: **{BENCHMARK_NAME}** (^NSEI)")
    st.caption(f"Min data: {MIN_DATA_POINTS} trading days")

    st.markdown("---")
    st.markdown("#### 📋 CSV Format")
    st.code("Ticker,Weight,AssetType\nRELIANCE.NS,15,Stock\nHDFC Top 100 Fund,10,MutualFund\nGOLDBEES.NS,5,ETF", language="text")
    st.caption("AssetType is optional — auto-detected from ticker.")

    st.markdown("---")
    st.markdown("#### 📡 Data Sources")
    st.markdown("""
<div style="font-size:13px;line-height:2.2;color:#4A5568">
  <span style="color:#3B5BDB;font-weight:700">●</span>&nbsp;
  <strong>yfinance</strong> — NSE/BSE stocks &amp; ETFs<br>
  <span style="color:#DD6B20;font-weight:700">●</span>&nbsp;
  <strong>mfapi.in</strong> — Mutual fund NAV (AMFI)<br>
  <span style="color:#E53E3E;font-weight:700">●</span>&nbsp;
  <strong>Unavailable</strong> — Flagged with reason
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 28px">
  <h1 style="font-size:30px;font-weight:700;color:#1A202C;margin:0;line-height:1.2">
    Portfolio Weighted β Calculator
  </h1>
  <p style="font-size:14px;color:#718096;margin:8px 0 0;font-weight:400;line-height:1.6">
    OLS regression against NIFTY 50 &nbsp;·&nbsp;
    Stocks, ETFs &amp; Mutual Funds &nbsp;·&nbsp;
    Indian market
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
col_up, col_btn = st.columns([4, 1])
with col_up:
    uploaded = st.file_uploader("Upload Portfolio CSV", type=["csv"],
                                label_visibility="collapsed")
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("Load Sample"):
        st.session_state["use_sample"] = True

csv_text = None
if uploaded:
    csv_text = uploaded.read().decode("utf-8", errors="ignore")
    st.session_state.pop("use_sample", None)
elif st.session_state.get("use_sample"):
    csv_text = SAMPLE_CSV

if csv_text:
    try:
        portfolio_df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()

    with st.expander("📂 Portfolio Preview", expanded=True):
        total_w = pd.to_numeric(portfolio_df.iloc[:, 1], errors="coerce").sum()
        st.markdown(
            f'<p style="font-size:13px;color:#38A169;font-weight:500;margin-bottom:8px">'
            f'✓ {len(portfolio_df)} holdings detected &nbsp;·&nbsp; '
            f'Total weight: {total_w:.2f} — will be normalized to 100%</p>',
            unsafe_allow_html=True
        )
        st.dataframe(portfolio_df, use_container_width=True, height=220)

    st.markdown("---")

    if st.button("⚡  Calculate Weighted Beta"):
        progress_bar = st.progress(0)
        status_text  = st.empty()
        t0 = time.time()

        def progress_cb(current, total, msg):
            pct = int((current / max(total, 1)) * 100)
            progress_bar.progress(pct)
            status_text.markdown(
                f'<p style="font-size:12px;color:#718096">{msg}</p>',
                unsafe_allow_html=True
            )

        try:
            pr = calculate_portfolio_beta(portfolio_df, lookback_years=lookback,
                                          progress_callback=progress_cb)
        except Exception as e:
            st.error(f"Calculation error: {e}")
            st.stop()

        progress_bar.progress(100)
        status_text.markdown(
            f'<p style="font-size:12px;color:#38A169;font-weight:500">'
            f'✓ Completed in {time.time()-t0:.1f}s</p>',
            unsafe_allow_html=True
        )
        st.session_state["result"] = pr


# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────
if "result" in st.session_state:
    pr = st.session_state["result"]
    rp_label, rp_color = risk_profile(pr.weighted_beta)
    df_out = results_to_dataframe(pr)

    st.markdown("---")
    st.markdown('<h2 style="font-size:20px;font-weight:700;color:#1A202C;margin-bottom:20px">Results</h2>',
                unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Portfolio β", f"{pr.weighted_beta:.4f}" if pr.weighted_beta else "N/A",
                  help="OLS beta vs NIFTY 50")
    with k2:
        st.metric("Risk Profile", rp_label)
    with k3:
        st.metric("Data Coverage", f"{pr.coverage_pct:.1f}%",
                  delta=f"{pr.available}/{pr.total_holdings} holdings",
                  delta_color="normal" if pr.coverage_pct > 80 else "inverse")
    with k4:
        st.metric("Unavailable", pr.unavailable,
                  delta="See warnings" if pr.unavailable else "Full coverage",
                  delta_color="inverse" if pr.unavailable else "normal")
    with k5:
        st.metric("Benchmark", BENCHMARK_NAME)

    st.caption(f"Calculated: {pr.calculation_date}  ·  Lookback: {lookback}Y  ·  Min data: {MIN_DATA_POINTS} pts")

    # Interpretation banner
    if pr.weighted_beta:
        if pr.weighted_beta < 1.0:
            txt    = f"Your portfolio (β = {pr.weighted_beta:.4f}) is <strong>less volatile than the NIFTY 50</strong>. It is expected to fall less during market downturns, and rise less during rallies."
            accent = "#38A169"; bg = "#F0FFF4"; border = "#C6F6D5"
        elif pr.weighted_beta == 1.0:
            txt    = "Your portfolio (β = 1.0) <strong>moves in line with the NIFTY 50</strong>."
            accent = "#3B5BDB"; bg = "#EEF2FF"; border = "#C7D2FE"
        else:
            txt    = f"Your portfolio (β = {pr.weighted_beta:.4f}) is <strong>more volatile than the NIFTY 50</strong>. Expect amplified gains in bull markets and steeper losses in downturns."
            accent = "#DD6B20"; bg = "#FFFAF0"; border = "#FEEBC8"

        st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-left:4px solid {accent};
            border-radius:10px;padding:14px 20px;margin:16px 0;font-size:14px;
            color:#1A202C;line-height:1.7">{txt}</div>
""", unsafe_allow_html=True)

    if pr.warnings:
        st.warning(
            f"⚠️ **{pr.unavailable} holding(s) had data issues** — beta computed over "
            f"{pr.coverage_pct:.1f}% of portfolio weight.\n\n" +
            "\n".join(f"• {w}" for w in pr.warnings)
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Holdings Detail", "📈 Visualizations",
        "⚖️ Contribution", "🛡️ Futures Hedge", "📥 Export"
    ])

    # ── Tab 1: Holdings ────────────────────────────────────────────────
    with tab1:
        st.markdown('<h3 style="font-size:16px;font-weight:600;color:#1A202C;margin:16px 0 12px">Individual Holding Betas</h3>', unsafe_allow_html=True)
        d = df_out.copy()
        d["Beta"]     = d["Beta"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "⚠ N/A")
        d["R²"]       = d["R-Squared"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        d["Risk"]     = df_out["Beta"].apply(lambda x: beta_label(x) if pd.notna(x) else "⚠ UNAVAILABLE")
        d["Weight %"] = d["Norm Weight (%)"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(
            d[["Ticker / Fund", "Asset Type", "Weight %", "Beta", "R²",
               "Risk", "Data Source", "Period Used", "Notes"]],
            use_container_width=True, height=400
        )

    # ── Tab 2: Visualizations ─────────────────────────────────────────
    with tab2:
        st.markdown('<h3 style="font-size:16px;font-weight:600;color:#1A202C;margin:16px 0 4px">Beta Distribution Across Holdings</h3>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:13px;color:#718096;margin-bottom:16px">Each bar shows the individual beta vs NIFTY 50. Colour indicates risk level.</p>', unsafe_allow_html=True)

        valid = [h for h in pr.holdings if h.beta is not None]
        if valid:
            fig_bar = go.Figure()
            for h in sorted(valid, key=lambda x: x.beta, reverse=True):
                fig_bar.add_trace(go.Bar(
                    x=[h.ticker], y=[h.beta],
                    marker_color=beta_color(h.beta),
                    text=[f"β={h.beta:.3f}"], textposition="outside",
                    name=h.ticker, showlegend=False,
                    hovertemplate=(
                        f"<b>{h.ticker}</b><br>Beta: {h.beta:.4f}<br>"
                        f"Weight: {h.norm_weight:.2f}%<br>"
                        f"Source: {h.data_source}<extra></extra>"
                    )
                ))
            fig_bar.add_hline(y=1.0, line_dash="dash", line_color="#CBD5E0",
                              annotation_text="Market (β=1)", annotation_position="right",
                              annotation_font_color="#A0AEC0")
            if pr.weighted_beta:
                fig_bar.add_hline(y=pr.weighted_beta, line_dash="dot", line_color="#3B5BDB",
                                  annotation_text=f"Portfolio β={pr.weighted_beta:.3f}",
                                  annotation_position="right",
                                  annotation_font_color="#3B5BDB")
            fig_bar.update_layout(
                **PLOT_BASE,
                xaxis=dict(tickangle=-30, gridcolor="#E2E8F0", title=""),
                yaxis=dict(gridcolor="#E2E8F0", title="Beta (vs NIFTY 50)"),
                height=420, margin=dict(t=24, b=60, l=48, r=130),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown('<h3 style="font-size:16px;font-weight:600;color:#1A202C;margin:24px 0 4px">Risk vs Allocation — β by Portfolio Weight</h3>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:13px;color:#718096;margin-bottom:16px">Bubble size represents portfolio weight. Holdings top-right drive the most risk.</p>', unsafe_allow_html=True)

            fig_sc = go.Figure()
            for h in valid:
                fig_sc.add_trace(go.Scatter(
                    x=[h.norm_weight], y=[h.beta],
                    mode="markers+text", text=[h.ticker], textposition="top center",
                    marker=dict(size=h.norm_weight * 2.5, color=beta_color(h.beta),
                                opacity=0.85, line=dict(width=1, color="#FFFFFF")),
                    showlegend=False, name=h.ticker,
                    hovertemplate=(f"<b>{h.ticker}</b><br>Weight: {h.norm_weight:.2f}%<br>"
                                   f"Beta: {h.beta:.4f}<extra></extra>")
                ))
            fig_sc.add_hline(y=1.0, line_dash="dash", line_color="#CBD5E0")
            fig_sc.update_layout(
                **PLOT_BASE,
                xaxis=dict(gridcolor="#E2E8F0", title="Portfolio Weight (%)"),
                yaxis=dict(gridcolor="#E2E8F0", title="Beta (vs NIFTY 50)"),
                height=420,
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.warning("No valid beta values to visualize.")

    # ── Tab 3: Contribution ───────────────────────────────────────────
    with tab3:
        st.markdown('<h3 style="font-size:16px;font-weight:600;color:#1A202C;margin:16px 0 4px">Weighted Beta Contribution by Holding</h3>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:13px;color:#718096;margin-bottom:16px">Each bar shows how much a holding contributes to the overall portfolio beta.</p>', unsafe_allow_html=True)

        valid = [h for h in pr.holdings if h.beta is not None]
        if valid and pr.weighted_beta:
            contribs = [
                (h.ticker, h.beta * h.norm_weight / pr.coverage_pct,
                 h.norm_weight, h.beta, h.data_source)
                for h in sorted(valid, key=lambda x: x.beta * x.norm_weight, reverse=True)
            ]
            fig_c = go.Figure(go.Bar(
                y=[c[0] for c in contribs], x=[c[1] for c in contribs],
                orientation="h",
                marker_color=[beta_color(c[3]) for c in contribs],
                text=[f"β={c[3]:.3f} × {c[2]:.1f}% = {c[1]:.4f}" for c in contribs],
                textposition="inside",
                insidetextfont=dict(color="#FFFFFF", size=11),
                hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
            ))
            fig_c.add_vline(x=pr.weighted_beta, line_dash="dot", line_color="#3B5BDB",
                            annotation_text=f"Portfolio β = {pr.weighted_beta:.3f}",
                            annotation_position="top right",
                            annotation_font_color="#3B5BDB")
            fig_c.update_layout(
                **PLOT_BASE,
                xaxis=dict(gridcolor="#E2E8F0", title="Beta Contribution"),
                yaxis=dict(gridcolor="#E2E8F0", title=""),
                height=max(300, len(contribs) * 40),
            )
            st.plotly_chart(fig_c, use_container_width=True)

            cdf = pd.DataFrame(contribs, columns=["Ticker", "β Contribution", "Weight %", "Beta", "Source"])
            cdf["β Contribution"] = cdf["β Contribution"].apply(lambda x: f"{x:.5f}")
            cdf["Beta"]           = cdf["Beta"].apply(lambda x: f"{x:.4f}")
            cdf["Weight %"]       = cdf["Weight %"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(cdf[["Ticker", "Weight %", "Beta", "β Contribution", "Source"]],
                         use_container_width=True, height=300)
        else:
            st.warning("Contribution breakdown requires at least one valid beta.")

    # ── Tab 4: Futures Hedge ──────────────────────────────────────────
    with tab4:
        st.markdown('<h3 style="font-size:16px;font-weight:600;color:#1A202C;margin:16px 0 4px">NIFTY Futures Hedge Calculator</h3>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:13px;color:#718096;margin-bottom:16px">Calculates how many NIFTY 50 futures contracts to short to hedge your portfolio against broad market moves.</p>', unsafe_allow_html=True)

        if not pr.weighted_beta:
            st.warning("Weighted beta unavailable — cannot compute hedge.")
        else:
            st.markdown('<p style="font-size:13px;font-weight:600;color:#1A202C;margin:0 0 10px">⚙️ Hedge Inputs</p>', unsafe_allow_html=True)
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                portfolio_value = st.number_input("Portfolio Value (₹)", min_value=10000,
                    value=1000000, step=10000, format="%d",
                    help="Total current market value in INR")
            with hc2:
                nifty_spot = st.number_input("NIFTY 50 Spot Price", min_value=1000.0,
                    value=22500.0, step=50.0, format="%.2f",
                    help="Current NIFTY 50 index level")
            with hc3:
                lot_size = st.number_input("NIFTY Lot Size", min_value=1, value=75, step=1,
                    help="Verify current lot size on NSE before trading")

            hc4, hc5 = st.columns(2)
            with hc4:
                hedge_pct = st.slider("Hedge Ratio (%)", 10, 100, 100, step=5,
                    help="100% = full hedge. 50% = hedge half the market risk.")
            with hc5:
                target_beta = st.number_input("Target Beta after hedge",
                    min_value=0.0, max_value=float(pr.weighted_beta),
                    value=0.0, step=0.05, format="%.2f",
                    help="0 = fully market-neutral.")

            st.markdown("---")

            contract_value     = nifty_spot * lot_size
            beta_to_hedge      = pr.weighted_beta - target_beta
            raw_contracts      = (portfolio_value * beta_to_hedge) / contract_value
            contracts_to_short = raw_contracts * (hedge_pct / 100)
            contracts_rounded  = round(contracts_to_short)
            hedge_value        = contracts_rounded * contract_value
            residual_beta      = pr.weighted_beta - (contracts_rounded * contract_value / portfolio_value)
            hedge_eff          = (
                (beta_to_hedge - max(residual_beta - target_beta, 0)) / beta_to_hedge * 100
                if beta_to_hedge > 0 else 100
            )

            st.markdown(f"""
<div style="background:#EEF2FF;border:1.5px solid #C7D2FE;border-left:5px solid #3B5BDB;
            border-radius:12px;padding:28px 32px;margin-bottom:24px">
  <div style="font-size:11px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;
              color:#718096;margin-bottom:8px">NIFTY Futures Contracts to Short</div>
  <div style="font-size:56px;font-weight:700;color:#3B5BDB;line-height:1;margin-bottom:10px">
    {contracts_rounded}
  </div>
  <div style="font-size:14px;color:#4A5568">
    {contracts_rounded} contracts &nbsp;×&nbsp; ₹{contract_value:,.0f} per contract
    &nbsp;=&nbsp; <strong>₹{hedge_value:,.0f} notional hedge</strong>
  </div>
</div>
""", unsafe_allow_html=True)

            r1, r2, r3, r4 = st.columns(4)
            with r1: st.metric("Current Portfolio β", f"{pr.weighted_beta:.4f}")
            with r2: st.metric("β After Hedge", f"{max(residual_beta,0):.4f}",
                               delta=f"{residual_beta-pr.weighted_beta:.4f}", delta_color="inverse")
            with r3: st.metric("Hedge Effectiveness", f"{hedge_eff:.1f}%")
            with r4: st.metric("Hedge Ratio Applied", f"{hedge_pct}%")

            st.markdown("---")
            st.markdown('<p style="font-size:14px;font-weight:600;color:#1A202C;margin:0 0 12px">📐 Step-by-Step Calculation</p>', unsafe_allow_html=True)
            st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;
            padding:24px 28px;font-size:13px;line-height:2.4;color:#1A202C">
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
              color:#3B5BDB;margin-bottom:14px">Formula</div>
  <div style="color:#4A5568;margin-bottom:18px;font-family:'JetBrains Mono',monospace;font-size:12px">
    Contracts = (Portfolio Value × β_to_hedge × Hedge%) / (NIFTY Spot × Lot Size)
  </div>
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#A0AEC0;margin-bottom:4px">Step 1 — Portfolio Beta</div>
  <div style="padding-left:16px;margin-bottom:14px">Weighted β = <span style="color:#3B5BDB;font-weight:600">{pr.weighted_beta:.4f}</span> <span style="color:#A0AEC0">(OLS vs NIFTY 50, {lookback}Y)</span></div>
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#A0AEC0;margin-bottom:4px">Step 2 — Beta to Hedge</div>
  <div style="padding-left:16px;margin-bottom:14px">{pr.weighted_beta:.4f} − {target_beta:.2f} = <span style="color:#3B5BDB;font-weight:600">{beta_to_hedge:.4f}</span></div>
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#A0AEC0;margin-bottom:4px">Step 3 — Contract Value</div>
  <div style="padding-left:16px;margin-bottom:14px">₹{nifty_spot:,.2f} × {lot_size} = <span style="color:#3B5BDB;font-weight:600">₹{contract_value:,.0f}</span> per contract</div>
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#A0AEC0;margin-bottom:4px">Step 4 — Raw Contracts</div>
  <div style="padding-left:16px;margin-bottom:14px">(₹{portfolio_value:,.0f} × {beta_to_hedge:.4f}) / ₹{contract_value:,.0f} = <span style="color:#DD6B20;font-weight:600">{raw_contracts:.4f}</span></div>
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#A0AEC0;margin-bottom:4px">Step 5 — Apply Hedge Ratio ({hedge_pct}%)</div>
  <div style="padding-left:16px;margin-bottom:14px">{raw_contracts:.4f} × {hedge_pct/100:.2f} = {contracts_to_short:.4f} → Rounded: <span style="color:#3B5BDB;font-weight:700">{contracts_rounded} contracts</span></div>
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#A0AEC0;margin-bottom:4px">Step 6 — Residual Beta</div>
  <div style="padding-left:16px">{pr.weighted_beta:.4f} − ({contracts_rounded} × ₹{contract_value:,.0f} / ₹{portfolio_value:,.0f}) = <span style="color:#3B5BDB;font-weight:700">{max(residual_beta,0):.4f}</span></div>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<p style="font-size:14px;font-weight:600;color:#1A202C;margin:0 0 6px">📋 Per-Holding Hedge Contribution</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:12px;color:#718096;margin-bottom:12px">Each holding\'s share of the total hedge requirement.</p>', unsafe_allow_html=True)

            valid_h = [h for h in pr.holdings if h.beta is not None]
            if valid_h:
                hrows = []
                for h in sorted(valid_h, key=lambda x: x.beta * x.norm_weight, reverse=True):
                    hv  = portfolio_value * h.norm_weight / 100
                    sbv = hv * h.beta
                    sc  = sbv / contract_value
                    sch = sc * (hedge_pct/100) * (beta_to_hedge/pr.weighted_beta if pr.weighted_beta else 1)
                    hrows.append({
                        "Ticker": h.ticker, "Type": h.asset_type,
                        "Holding Value (₹)": f"₹{hv:,.0f}", "Weight %": f"{h.norm_weight:.2f}%",
                        "Beta (β)": f"{h.beta:.4f}", "β × Value (₹)": f"₹{sbv:,.0f}",
                        "Frac. Contracts": f"{sc:.4f}", "Hedge Contribution": f"{sch:.4f}",
                        "Data Source": h.data_source,
                    })
                st.dataframe(pd.DataFrame(hrows), use_container_width=True, height=360)

                tbv  = sum(portfolio_value * h.norm_weight / 100 * h.beta for h in valid_h)
                tfrac = tbv / contract_value
                st.markdown(f"""
<div style="background:#F0FFF4;border:1px solid #C6F6D5;border-radius:10px;
            padding:14px 20px;font-size:13px;color:#1A202C;margin-top:8px">
  <span style="color:#718096">Total β-adjusted exposure:</span> <strong>₹{tbv:,.0f}</strong>
  &nbsp;·&nbsp;
  <span style="color:#718096">Total fractional contracts:</span> <strong>{tfrac:.4f}</strong>
  &nbsp;·&nbsp;
  <span style="color:#718096">Contracts to SHORT:</span>
  <span style="color:#3B5BDB;font-weight:700;font-size:15px"> {contracts_rounded}</span>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<p style="font-size:14px;font-weight:600;color:#1A202C;margin:0 0 6px">📊 Hedge Scenario Matrix</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:12px;color:#718096;margin-bottom:12px">Contracts needed across different hedge ratios and NIFTY index levels.</p>', unsafe_allow_html=True)
            nlevels = [nifty_spot * m for m in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]]
            sdata   = {"NIFTY Level": [f"₹{n:,.0f}" for n in nlevels]}
            for hp in [25, 50, 75, 100]:
                sdata[f"{hp}% Hedge"] = [
                    f"{round((portfolio_value * beta_to_hedge)/(ns*lot_size)*hp/100)} lots"
                    for ns in nlevels
                ]
            st.dataframe(pd.DataFrame(sdata), use_container_width=True, height=290)

            st.markdown("""
<div style="background:#FFFAF0;border:1px solid #FEEBC8;border-left:4px solid #DD6B20;
            border-radius:10px;padding:14px 20px;font-size:12px;color:#744210;
            line-height:1.9;margin-top:16px">
  <strong>⚠ Important:</strong> This hedge quantity is based on <em>historical beta</em> and is an approximation.
  Actual effectiveness depends on realized correlation, execution slippage, roll costs, and basis risk.
  The lot size changes periodically — always verify on NSE before placing orders.
  Futures trading involves margin requirements. <strong>This is not financial advice.</strong>
</div>
""", unsafe_allow_html=True)

    # ── Tab 5: Export ─────────────────────────────────────────────────
    with tab5:
        st.markdown('<h3 style="font-size:16px;font-weight:600;color:#1A202C;margin:16px 0 12px">Export Results</h3>', unsafe_allow_html=True)
        st.download_button(
            label="⬇ Download Full Results CSV",
            data=df_out.to_csv(index=False),
            file_name=f"portfolio_beta_{pr.calculation_date[:10]}.csv",
            mime="text/csv",
        )
        st.markdown("---")
        st.markdown('<p style="font-size:14px;font-weight:600;color:#1A202C;margin:0 0 12px">Summary</p>', unsafe_allow_html=True)
        st.markdown(f"""
| Field | Value |
|---|---|
| Portfolio Weighted Beta | **{pr.weighted_beta:.4f}** |
| Risk Profile | **{rp_label}** |
| Benchmark | {BENCHMARK_NAME} |
| Data Coverage | {pr.coverage_pct:.1f}% ({pr.available}/{pr.total_holdings} holdings) |
| Calculation Date | {pr.calculation_date} |
| Lookback Period | {lookback} year(s) |
""")

    # Methodology
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 Methodology & Disclaimers"):
        st.markdown(f"""
**Beta Calculation**

OLS regression of daily log-returns of each asset vs NIFTY 50 (`^NSEI`).
Formula: `β = Cov(Rᵢ, Rₘ) / Var(Rₘ)` — computed via `scipy.stats.linregress`.

**Data Sources**
- **yfinance** — {lookback}Y of daily adjusted closes for NSE/BSE stocks and ETFs. Tickers are auto-suffixed with `.NS` (NSE) or `.BO` (BSE).
- **mfapi.in** — Free AMFI-backed API for mutual fund NAV history. Fund names are fuzzy-matched against the AMFI scheme list.

**Weighted Beta**

`Portfolio β = Σ (βᵢ × wᵢ) / Σ wᵢ` — computed only over holdings with valid beta.

**Why a holding may be unavailable**
- Ticker not found on Yahoo Finance (delisted, OTC, or spelling error)
- Insufficient price history (fewer than {MIN_DATA_POINTS} trading days)
- Mutual fund name not matched in AMFI list
- mfapi.in rate limit or network error

> ⚠ For informational purposes only. Beta is a historical measure and does not predict future risk.
> Consult a SEBI-registered advisor before making investment decisions.
""")

# ─────────────────────────────────────────────
# Landing state
# ─────────────────────────────────────────────
else:
    st.markdown("""
<div style="border:2px dashed #CBD5E0;border-radius:16px;padding:64px 40px;
            text-align:center;background:#FFFFFF;margin-top:16px">
  <div style="font-size:52px;color:#3B5BDB;font-weight:700;line-height:1;margin-bottom:16px">β</div>
  <div style="font-size:22px;font-weight:700;color:#1A202C;margin-bottom:10px">
    Upload your portfolio CSV to begin
  </div>
  <div style="font-size:14px;color:#718096;line-height:2;max-width:500px;margin:0 auto">
    Required columns:
    <span style="color:#3B5BDB;font-weight:600">Ticker</span>,
    <span style="color:#3B5BDB;font-weight:600">Weight</span>,
    <span style="color:#3B5BDB;font-weight:600">AssetType</span>
    <br>
    Supports NSE stocks (<code>RELIANCE.NS</code>), BSE stocks (<code>RELIANCE.BO</code>),
    ETFs (<code>GOLDBEES.NS</code>), and Mutual Funds by name (<code>HDFC Top 100 Fund</code>)
    <br><br>
    Or click <strong>Load Sample</strong> to try a pre-built Indian portfolio instantly.
  </div>
</div>
""", unsafe_allow_html=True)
