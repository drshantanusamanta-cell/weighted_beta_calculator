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
import plotly.express as px
import streamlit as st

from beta_engine import (
    calculate_portfolio_beta,
    results_to_dataframe,
    BENCHMARK_NAME,
    LOOKBACK_YEARS,
    MIN_DATA_POINTS,
)

# ─────────────────────────────────────────────
# Page config & custom CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BetaCalc — Portfolio Beta Engine",
    page_icon="β",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Sora:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0A0E1A;
    color: #E2E8F0;
}
h1, h2, h3 { font-family: 'Sora', sans-serif !important; }

/* Main background */
.stApp { background-color: #0A0E1A; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0F1629;
    border-right: 1px solid #1E2D40;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1E2D40;
    border-radius: 8px;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace;
    font-size: 10px !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748B !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif;
    font-size: 28px !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background: #00FFB2 !important;
    color: #0A0E1A !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-size: 12px !important;
    padding: 10px 24px !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1E2D40 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

/* Progress */
.stProgress > div > div { background-color: #00FFB2 !important; }

/* Warning / info boxes */
.stAlert { border-radius: 6px !important; font-family: 'DM Mono', monospace !important; font-size: 12px !important; }

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: #111827 !important;
    border: 1.5px dashed #1E2D40 !important;
    border-radius: 8px !important;
}

/* Selectbox / slider */
[data-baseweb="select"] { font-family: 'DM Mono', monospace !important; }
.stSlider [data-baseweb="slider"] [role="slider"] { background: #00FFB2 !important; }
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMin"],
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMax"] {
    font-family: 'DM Mono', monospace !important; font-size: 11px !important;
}

/* Tabs */
[data-baseweb="tab"] { font-family: 'DM Mono', monospace !important; font-size: 11px !important; letter-spacing: 1px; }
[aria-selected="true"] { border-bottom-color: #00FFB2 !important; color: #00FFB2 !important; }

/* Divider */
hr { border-color: #1E2D40 !important; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
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

def beta_color(beta):
    if beta is None: return "#64748B"
    if beta < 0.7:   return "#10B981"
    if beta < 1.2:   return "#00FFB2"
    if beta < 1.8:   return "#F59E0B"
    return "#EF4444"

def beta_label(beta):
    if beta is None: return "N/A"
    if beta < 0.7:   return "🟢 Low Risk"
    if beta < 1.2:   return "🟡 Moderate"
    if beta < 1.8:   return "🟠 Aggressive"
    return "🔴 Very High"

def risk_profile(wb):
    if wb is None: return "Unknown", "#64748B"
    if wb < 0.6:   return "Conservative",           "#10B981"
    if wb < 0.9:   return "Moderate-Conservative",  "#34D399"
    if wb < 1.1:   return "Market-Neutral",         "#00FFB2"
    if wb < 1.4:   return "Aggressive Growth",      "#F59E0B"
    return         "High Risk / Speculative",        "#EF4444"

def source_badge(src):
    colors = {
        "yfinance":    ("#00FFB2", "Live — Yahoo Finance"),
        "mfapi":       ("#F59E0B", "Live — AMFI/mfapi.in"),
        "unavailable": ("#EF4444", "Unavailable"),
    }
    return colors.get(src, ("#64748B", src))


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## **β** BetaCalc")
    st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#64748B;text-transform:uppercase">Weighted Portfolio Beta Engine</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### ⚙️ Settings")
    lookback = st.slider("Lookback Period (years)", 1, 5, LOOKBACK_YEARS,
                         help="Historical window for OLS regression vs NIFTY 50")
    st.caption(f"Benchmark: **{BENCHMARK_NAME}** (^NSEI)")
    st.caption(f"Min data points: {MIN_DATA_POINTS} trading days")

    st.markdown("---")
    st.markdown("#### 📋 CSV Format")
    st.code("Ticker,Weight,AssetType\nRELIANCE.NS,15,Stock\nHDFC Top 100 Fund,10,MutualFund\nGOLDBEES.NS,5,ETF", language="text")
    st.caption("AssetType column is optional — auto-detected from ticker name.")

    st.markdown("---")
    st.markdown("#### 📡 Data Sources")
    st.markdown("""
<div style='font-size:11px;line-height:1.9;color:#94A3B8'>
<span style='color:#00FFB2'>●</span> <b>yfinance</b><br>
&nbsp;&nbsp;NSE/BSE stocks + ETFs<br>
&nbsp;&nbsp;(.NS / .BO suffixes)<br><br>
<span style='color:#F59E0B'>●</span> <b>mfapi.in</b><br>
&nbsp;&nbsp;Mutual fund NAV history<br>
&nbsp;&nbsp;via AMFI scheme codes<br><br>
<span style='color:#EF4444'>●</span> <b>Unavailable</b><br>
&nbsp;&nbsp;Clearly flagged with reason
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.markdown('<h1 style="font-family:Sora;font-size:32px;font-weight:800;letter-spacing:-1px;margin-bottom:4px">Portfolio Weighted β Calculator</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#64748B;margin-bottom:32px">Real OLS Beta · yfinance + AMFI mfapi · NIFTY 50 Benchmark</p>', unsafe_allow_html=True)

# ── Upload section ─────────────────────────────────────────────────────
col_up, col_sample = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader("Upload Portfolio CSV", type=["csv"],
                                 label_visibility="collapsed")
with col_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Load Sample"):
        st.session_state["use_sample"] = True

# Determine source CSV
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

    # ── Preview ──────────────────────────────────────────────────────
    with st.expander("📂 Portfolio Preview", expanded=True):
        total_w = pd.to_numeric(portfolio_df.iloc[:, 1], errors="coerce").sum()
        st.markdown(f'<p style="font-size:12px;color:#00FFB2">✓ {len(portfolio_df)} holdings detected &nbsp;|&nbsp; Total weight: {total_w:.2f} (will normalize to 100%)</p>', unsafe_allow_html=True)
        st.dataframe(portfolio_df, use_container_width=True, height=220)

    st.markdown("---")

    # ── Calculate button ──────────────────────────────────────────────
    if st.button("⚡  Calculate Weighted Beta"):
        progress_bar = st.progress(0)
        status_text  = st.empty()
        start_time   = time.time()

        def progress_cb(current, total, msg):
            pct = int((current / max(total, 1)) * 100)
            progress_bar.progress(pct)
            status_text.markdown(
                f'<p style="font-size:11px;color:#64748B;letter-spacing:1px">{msg}</p>',
                unsafe_allow_html=True
            )

        try:
            pr = calculate_portfolio_beta(
                portfolio_df,
                lookback_years=lookback,
                progress_callback=progress_cb,
            )
        except Exception as e:
            st.error(f"Calculation error: {e}")
            st.stop()

        progress_bar.progress(100)
        elapsed = time.time() - start_time
        status_text.markdown(
            f'<p style="font-size:11px;color:#00FFB2">✓ Done in {elapsed:.1f}s</p>',
            unsafe_allow_html=True
        )

        st.session_state["result"] = pr

# ── Results ──────────────────────────────────────────────────────────
if "result" in st.session_state:
    pr = st.session_state["result"]
    rp_label, rp_color = risk_profile(pr.weighted_beta)
    df_out = results_to_dataframe(pr)

    st.markdown("---")
    st.markdown('<h2 style="font-family:Sora;font-size:20px;font-weight:700;margin-bottom:20px">Results</h2>', unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        wb = f"{pr.weighted_beta:.4f}" if pr.weighted_beta else "N/A"
        st.metric("Portfolio β (Weighted)", wb, help="OLS beta of portfolio vs NIFTY 50")
    with k2:
        st.metric("Risk Profile", rp_label)
    with k3:
        st.metric("Data Coverage", f"{pr.coverage_pct:.1f}%",
                  delta=f"{pr.available}/{pr.total_holdings} holdings",
                  delta_color="normal" if pr.coverage_pct > 80 else "inverse")
    with k4:
        st.metric("Unavailable", pr.unavailable,
                  delta="See warnings below" if pr.unavailable else "Full coverage",
                  delta_color="inverse" if pr.unavailable else "normal")
    with k5:
        st.metric("Benchmark", BENCHMARK_NAME)

    st.caption(f"Calculated: {pr.calculation_date}  |  Lookback: {lookback}Y  |  Min data: {MIN_DATA_POINTS} pts")

    # ── Warnings ───────────────────────────────────────────────────────
    if pr.warnings:
        st.warning(f"⚠ **{pr.unavailable} holding(s) had data issues** — beta computed over {pr.coverage_pct:.1f}% of portfolio weight.\n\n" +
                   "\n".join(f"• {w}" for w in pr.warnings))

    # ── Tabs ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Holdings Detail", "📈 Visualizations", "⚖️ Contribution", "🛡️ Futures Hedge", "📥 Export"])

    with tab1:
        st.markdown("#### Individual Holding Betas")

        # Styled display dataframe
        display_df = df_out.copy()
        display_df["Beta"] = display_df["Beta"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "⚠ N/A"
        )
        display_df["R²"] = display_df["R-Squared"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "—"
        )
        display_df["Risk"] = df_out["Beta"].apply(
            lambda x: beta_label(x) if pd.notna(x) else "⚠ UNAVAILABLE"
        )
        display_df["Weight %"] = display_df["Norm Weight (%)"].apply(lambda x: f"{x:.2f}%")

        show_cols = ["Ticker / Fund", "Asset Type", "Weight %", "Beta", "R²",
                     "Risk", "Data Source", "Period Used", "Notes"]
        st.dataframe(
            display_df[show_cols],
            use_container_width=True,
            height=400,
        )

    with tab2:
        st.markdown("#### Beta Distribution Across Holdings")

        valid = [h for h in pr.holdings if h.beta is not None]
        if valid:
            fig_bar = go.Figure()
            for h in sorted(valid, key=lambda x: x.beta, reverse=True):
                color = beta_color(h.beta)
                fig_bar.add_trace(go.Bar(
                    x=[h.ticker],
                    y=[h.beta],
                    marker_color=color,
                    text=[f"β={h.beta:.3f}"],
                    textposition="outside",
                    name=h.ticker,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{h.ticker}</b><br>"
                        f"Beta: {h.beta:.4f}<br>"
                        f"Weight: {h.norm_weight:.2f}%<br>"
                        f"Source: {h.data_source}<br>"
                        f"R²: {h.r_squared:.3f}<extra></extra>"
                    )
                ))

            fig_bar.add_hline(y=1.0, line_dash="dash", line_color="#64748B",
                              annotation_text="Market (β=1)", annotation_position="right")
            if pr.weighted_beta:
                fig_bar.add_hline(y=pr.weighted_beta, line_dash="dot", line_color="#00FFB2",
                                  annotation_text=f"Portfolio β={pr.weighted_beta:.3f}",
                                  annotation_position="right")

            fig_bar.update_layout(
                paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
                font=dict(family="DM Mono", color="#94A3B8", size=11),
                xaxis=dict(tickangle=-30, gridcolor="#1E2D40", title=""),
                yaxis=dict(gridcolor="#1E2D40", title="Beta (vs NIFTY 50)"),
                height=420,
                margin=dict(t=20, b=60, l=40, r=120),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Scatter: Beta vs Weight
            st.markdown("#### Risk-Weight Scatter — β vs Portfolio Allocation")
            fig_sc = go.Figure()
            for h in valid:
                fig_sc.add_trace(go.Scatter(
                    x=[h.norm_weight],
                    y=[h.beta],
                    mode="markers+text",
                    text=[h.ticker],
                    textposition="top center",
                    marker=dict(size=h.norm_weight * 2.5,
                                color=beta_color(h.beta),
                                opacity=0.85,
                                line=dict(width=1, color="#0A0E1A")),
                    showlegend=False,
                    name=h.ticker,
                    hovertemplate=(
                        f"<b>{h.ticker}</b><br>Weight: {h.norm_weight:.2f}%<br>"
                        f"Beta: {h.beta:.4f}<extra></extra>"
                    )
                ))
            fig_sc.add_hline(y=1.0, line_dash="dash", line_color="#64748B")
            fig_sc.update_layout(
                paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
                font=dict(family="DM Mono", color="#94A3B8", size=11),
                xaxis=dict(gridcolor="#1E2D40", title="Portfolio Weight (%)"),
                yaxis=dict(gridcolor="#1E2D40", title="Beta (vs NIFTY 50)"),
                height=420,
                margin=dict(t=20, b=40, l=40, r=20),
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.warning("No valid beta values to visualize.")

    with tab3:
        st.markdown("#### Weighted Beta Contribution by Holding")
        valid = [h for h in pr.holdings if h.beta is not None]
        if valid and pr.weighted_beta:
            contribs = [(h.ticker, h.beta * h.norm_weight / pr.coverage_pct,
                         h.norm_weight, h.beta, h.data_source)
                        for h in sorted(valid, key=lambda x: x.beta * x.norm_weight, reverse=True)]

            fig_c = go.Figure(go.Bar(
                y=[c[0] for c in contribs],
                x=[c[1] for c in contribs],
                orientation="h",
                marker_color=[beta_color(c[3]) for c in contribs],
                text=[f"β={c[3]:.3f} × {c[2]:.1f}% = {c[1]:.4f}" for c in contribs],
                textposition="inside",
                hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
            ))
            fig_c.add_vline(x=pr.weighted_beta, line_dash="dot", line_color="#00FFB2",
                            annotation_text=f"Portfolio β={pr.weighted_beta:.3f}",
                            annotation_position="top right")
            fig_c.update_layout(
                paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
                font=dict(family="DM Mono", color="#94A3B8", size=11),
                xaxis=dict(gridcolor="#1E2D40", title="Beta Contribution"),
                yaxis=dict(gridcolor="#1E2D40", title=""),
                height=max(300, len(contribs) * 36),
                margin=dict(t=20, b=40, l=40, r=20),
            )
            st.plotly_chart(fig_c, use_container_width=True)

            # Summary table
            contrib_df = pd.DataFrame(contribs, columns=["Ticker", "β Contribution", "Weight %", "Beta", "Source"])
            contrib_df["β Contribution"] = contrib_df["β Contribution"].apply(lambda x: f"{x:.5f}")
            contrib_df["Beta"]           = contrib_df["Beta"].apply(lambda x: f"{x:.4f}")
            contrib_df["Weight %"]       = contrib_df["Weight %"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(contrib_df[["Ticker", "Weight %", "Beta", "β Contribution", "Source"]],
                         use_container_width=True, height=300)
        else:
            st.warning("Contribution breakdown requires at least one valid beta.")

    # ── TAB 4: FUTURES HEDGE ──────────────────────────────────────────
    with tab4:
        st.markdown("#### 🛡️ NIFTY Futures Hedge Calculator")
        st.markdown('<p style="font-size:12px;color:#64748B">Computes how many NIFTY 50 futures contracts to SHORT to fully or partially hedge your portfolio against market risk.</p>', unsafe_allow_html=True)

        if not pr.weighted_beta:
            st.warning("Weighted beta unavailable — cannot compute hedge. Ensure at least one holding has valid beta data.")
        else:
            st.markdown("---")

            # ── Inputs ────────────────────────────────────────────────
            st.markdown("##### ⚙️ Hedge Inputs")
            hcol1, hcol2, hcol3 = st.columns(3)

            with hcol1:
                portfolio_value = st.number_input(
                    "Portfolio Value (₹)",
                    min_value=10000,
                    value=1000000,
                    step=10000,
                    format="%d",
                    help="Total current market value of your portfolio in INR",
                )
            with hcol2:
                nifty_spot = st.number_input(
                    "NIFTY 50 Spot Price",
                    min_value=1000.0,
                    value=22500.0,
                    step=50.0,
                    format="%.2f",
                    help="Current NIFTY 50 index level (check NSE/Zerodha)",
                )
            with hcol3:
                lot_size = st.number_input(
                    "NIFTY Lot Size",
                    min_value=1,
                    value=65,
                    step=1,
                    help="Current NIFTY futures lot size (65 as of 2025; verify on NSE)",
                )

            hcol4, hcol5 = st.columns(2)
            with hcol4:
                hedge_pct = st.slider(
                    "Hedge Ratio (%)",
                    min_value=10, max_value=100, value=100, step=5,
                    help="100% = full hedge. 50% = hedge only half the market risk.",
                )
            with hcol5:
                target_beta = st.number_input(
                    "Target Portfolio Beta (after hedge)",
                    min_value=0.0,
                    max_value=float(pr.weighted_beta),
                    value=0.0,
                    step=0.05,
                    format="%.2f",
                    help="0 = fully market-neutral. 0.5 = half the market risk remains.",
                )

            st.markdown("---")

            # ── Core calculations ─────────────────────────────────────
            contract_value      = nifty_spot * lot_size
            beta_to_hedge       = pr.weighted_beta - target_beta
            raw_contracts       = (portfolio_value * beta_to_hedge) / contract_value
            contracts_to_short  = raw_contracts * (hedge_pct / 100)
            contracts_rounded   = round(contracts_to_short)
            hedge_value         = contracts_rounded * contract_value
            residual_beta       = pr.weighted_beta - (contracts_rounded * contract_value / portfolio_value)
            hedge_effectiveness = (beta_to_hedge - max(residual_beta - target_beta, 0)) / beta_to_hedge * 100 if beta_to_hedge > 0 else 100

            # ── Result Banner ─────────────────────────────────────────
            st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(0,255,178,0.08),rgba(0,255,178,0.03));
            border:1.5px solid rgba(0,255,178,0.4);border-radius:10px;padding:28px 32px;margin-bottom:24px">
  <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#64748B;margin-bottom:8px">
    NIFTY FUTURES CONTRACTS TO SHORT
  </div>
  <div style="font-family:'Sora',sans-serif;font-size:56px;font-weight:800;color:#00FFB2;letter-spacing:-2px;line-height:1">
    {contracts_rounded}
  </div>
  <div style="font-size:13px;color:#94A3B8;margin-top:8px">
    contracts &nbsp;×&nbsp; ₹{contract_value:,.0f}/contract &nbsp;=&nbsp;
    <span style="color:#E2E8F0;font-weight:600">₹{hedge_value:,.0f} notional hedge</span>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── KPI summary ───────────────────────────────────────────
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Portfolio β (Current)", f"{pr.weighted_beta:.4f}")
            with r2:
                st.metric("β After Hedge", f"{max(residual_beta, 0):.4f}",
                          delta=f"{residual_beta - pr.weighted_beta:.4f}",
                          delta_color="inverse")
            with r3:
                st.metric("Hedge Effectiveness", f"{hedge_effectiveness:.1f}%")
            with r4:
                st.metric("Hedge Ratio Applied", f"{hedge_pct}%")

            st.markdown("---")

            # ── Step-by-step methodology ──────────────────────────────
            st.markdown("##### 📐 Step-by-Step Calculation")

            st.markdown(f"""
<div style="background:#111827;border:1px solid #1E2D40;border-radius:8px;padding:24px;font-size:12px;line-height:2.2">

<div style="color:#00FFB2;font-weight:700;letter-spacing:1px;margin-bottom:12px">FORMULA</div>
<div style="color:#E2E8F0">
  Contracts to Short &nbsp;=&nbsp;
  <span style="color:#00FFB2">(Portfolio Value × β_to_hedge × Hedge%) / (NIFTY Spot × Lot Size)</span>
</div>
<br>

<div style="color:#64748B;letter-spacing:1px;font-size:10px;text-transform:uppercase;margin-bottom:8px">STEP 1 — Portfolio Beta</div>
<div style="color:#E2E8F0;padding-left:16px">
  Weighted Portfolio β &nbsp;=&nbsp; <span style="color:#00FFB2">{pr.weighted_beta:.4f}</span>
  &nbsp;&nbsp;<span style="color:#64748B">(OLS regression vs NIFTY 50, {lookback}Y lookback)</span>
</div>
<br>

<div style="color:#64748B;letter-spacing:1px;font-size:10px;text-transform:uppercase;margin-bottom:8px">STEP 2 — Beta to Hedge</div>
<div style="color:#E2E8F0;padding-left:16px">
  β_to_hedge &nbsp;=&nbsp; Current β − Target β
  &nbsp;=&nbsp; {pr.weighted_beta:.4f} − {target_beta:.2f}
  &nbsp;=&nbsp; <span style="color:#00FFB2">{beta_to_hedge:.4f}</span>
</div>
<br>

<div style="color:#64748B;letter-spacing:1px;font-size:10px;text-transform:uppercase;margin-bottom:8px">STEP 3 — Contract Value</div>
<div style="color:#E2E8F0;padding-left:16px">
  NIFTY Spot × Lot Size &nbsp;=&nbsp; ₹{nifty_spot:,.2f} × {lot_size}
  &nbsp;=&nbsp; <span style="color:#00FFB2">₹{contract_value:,.0f}</span> per contract
</div>
<br>

<div style="color:#64748B;letter-spacing:1px;font-size:10px;text-transform:uppercase;margin-bottom:8px">STEP 4 — Raw Contracts (before rounding & hedge %)</div>
<div style="color:#E2E8F0;padding-left:16px">
  (₹{portfolio_value:,.0f} × {beta_to_hedge:.4f}) / ₹{contract_value:,.0f}
  &nbsp;=&nbsp; <span style="color:#F59E0B">{raw_contracts:.4f} contracts</span>
</div>
<br>

<div style="color:#64748B;letter-spacing:1px;font-size:10px;text-transform:uppercase;margin-bottom:8px">STEP 5 — Apply Hedge Ratio ({hedge_pct}%)</div>
<div style="color:#E2E8F0;padding-left:16px">
  {raw_contracts:.4f} × {hedge_pct/100:.2f}
  &nbsp;=&nbsp; <span style="color:#F59E0B">{contracts_to_short:.4f} contracts</span>
  &nbsp;→ Rounded to nearest whole lot: <span style="color:#00FFB2;font-weight:700">{contracts_rounded} contracts</span>
</div>
<br>

<div style="color:#64748B;letter-spacing:1px;font-size:10px;text-transform:uppercase;margin-bottom:8px">STEP 6 — Residual Beta After Hedge</div>
<div style="color:#E2E8F0;padding-left:16px">
  β_residual &nbsp;=&nbsp; Current β − (Contracts × Contract Value / Portfolio Value)<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp; {pr.weighted_beta:.4f} − ({contracts_rounded} × ₹{contract_value:,.0f} / ₹{portfolio_value:,.0f})<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp; <span style="color:#00FFB2;font-weight:700">{max(residual_beta,0):.4f}</span>
</div>

</div>
""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Per-scrip hedge contribution table ────────────────────
            st.markdown("##### 📋 Per-Holding Hedge Contribution")
            st.markdown('<p style="font-size:11px;color:#64748B">Shows each holding\'s individual contribution to the total hedge requirement.</p>', unsafe_allow_html=True)

            valid_h = [h for h in pr.holdings if h.beta is not None]
            if valid_h:
                hedge_rows = []
                for h in sorted(valid_h, key=lambda x: x.beta * x.norm_weight, reverse=True):
                    holding_val      = portfolio_value * h.norm_weight / 100
                    scrip_beta_val   = holding_val * h.beta               # ₹ beta-adjusted exposure
                    scrip_contracts  = scrip_beta_val / contract_value    # fractional contracts
                    scrip_contracts_hedged = scrip_contracts * (hedge_pct / 100) * (beta_to_hedge / pr.weighted_beta if pr.weighted_beta else 1)
                    hedge_rows.append({
                        "Ticker":                h.ticker,
                        "Type":                  h.asset_type,
                        "Holding Value (₹)":     f"₹{holding_val:>12,.0f}",
                        "Weight %":              f"{h.norm_weight:.2f}%",
                        "Beta (β)":              f"{h.beta:.4f}",
                        "β × Value (₹)":         f"₹{scrip_beta_val:>12,.0f}",
                        "Frac. Contracts":       f"{scrip_contracts:.4f}",
                        "Hedge Contribution":    f"{scrip_contracts_hedged:.4f}",
                        "Data Source":           h.data_source,
                    })

                hedge_df = pd.DataFrame(hedge_rows)
                st.dataframe(hedge_df, use_container_width=True, height=400)

                # Totals row annotation
                total_beta_val = sum(portfolio_value * h.norm_weight / 100 * h.beta for h in valid_h)
                total_frac     = total_beta_val / contract_value
                st.markdown(f"""
<div style="background:rgba(0,255,178,0.05);border:1px solid rgba(0,255,178,0.2);
            border-radius:6px;padding:14px 18px;font-size:12px;margin-top:8px">
  <span style="color:#64748B">Total β-adjusted exposure: </span>
  <span style="color:#E2E8F0;font-weight:600">₹{total_beta_val:,.0f}</span>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <span style="color:#64748B">Total fractional contracts: </span>
  <span style="color:#E2E8F0;font-weight:600">{total_frac:.4f}</span>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <span style="color:#64748B">Final rounded contracts to SHORT: </span>
  <span style="color:#00FFB2;font-weight:700;font-size:14px">{contracts_rounded}</span>
</div>
""", unsafe_allow_html=True)

            # ── Hedge scenarios table ─────────────────────────────────
            st.markdown("---")
            st.markdown("##### 📊 Hedge Scenario Matrix")
            st.markdown('<p style="font-size:11px;color:#64748B">Contracts required at different hedge ratios and NIFTY levels.</p>', unsafe_allow_html=True)

            nifty_levels = [nifty_spot * m for m in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]]
            hedge_ratios  = [25, 50, 75, 100]
            scenario_data = {"NIFTY Level": [f"₹{n:,.0f}" for n in nifty_levels]}
            for hr_pct in hedge_ratios:
                col_contracts = []
                for ns in nifty_levels:
                    cv  = ns * lot_size
                    raw = (portfolio_value * beta_to_hedge) / cv
                    col_contracts.append(f"{round(raw * hr_pct / 100)} lots")
                scenario_data[f"{hr_pct}% Hedge"] = col_contracts

            st.dataframe(pd.DataFrame(scenario_data), use_container_width=True, height=290)

            # ── Disclaimer ────────────────────────────────────────────
            st.markdown("""
<div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.3);
            border-radius:6px;padding:14px 18px;font-size:11px;color:#FCD34D;margin-top:16px;line-height:1.8">
⚠ <b>Important:</b> This hedge quantity is based on <i>historical beta</i> and is an approximation.
Actual hedge effectiveness depends on realized correlation, execution slippage, roll costs,
and basis risk between portfolio and NIFTY futures. The lot size changes periodically — always
verify the current lot size on NSE (nseindia.com) before placing orders.
Futures hedging involves margin requirements — ensure adequate margin in your account.
This is not financial advice.
</div>
""", unsafe_allow_html=True)

    # ── TAB 5: EXPORT ─────────────────────────────────────────────────
    with tab5:
        st.markdown("#### Export Results")
        csv_export = df_out.to_csv(index=False)
        st.download_button(
            label="⬇ Download Full Results CSV",
            data=csv_export,
            file_name=f"portfolio_beta_{pr.calculation_date[:10]}.csv",
            mime="text/csv",
        )
        st.markdown("##### Summary")
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
    with st.expander("📖 Methodology & Disclaimers"):
        st.markdown(f"""
**Beta Calculation:**  
OLS regression of daily log-returns of each asset vs NIFTY 50 (^NSEI).  
`β = Cov(Rᵢ, Rₘ) / Var(Rₘ)` — computed via `scipy.stats.linregress`.

**Data Sources:**  
- **yfinance** — Fetches {lookback}Y of daily adjusted close prices for NSE/BSE stocks and ETFs.  
  Tickers automatically suffixed with `.NS` (NSE) or `.BO` (BSE) as fallback.  
- **mfapi.in** — Free AMFI-backed API for mutual fund NAV history.  
  Fund names are fuzzy-matched against the AMFI scheme list using token overlap.  

**Weighted Beta:**  
`Portfolio β = Σ (βᵢ × wᵢ) / Σ wᵢ` — computed only over holdings with valid beta.  
If some holdings are unavailable, coverage % is reported and portfolio beta is adjusted.

**Unavailability Reasons:**  
- Ticker not found on Yahoo Finance (delisted, OTC, spelling error)  
- Insufficient price history (< {MIN_DATA_POINTS} trading days)  
- Mutual fund name not matched in AMFI list (try AMFI-exact names)  
- mfapi.in rate limit or network error

> ⚠ This tool is for informational purposes only. Beta is a historical measure and does not guarantee future risk. Mutual fund betas may exhibit lag due to daily NAV reporting. Consult a SEBI-registered advisor before making investment decisions.
""")

else:
    # Landing state
    st.markdown("""
<div style='border:1.5px dashed #1E2D40;border-radius:8px;padding:60px 40px;text-align:center;background:rgba(255,255,255,0.01)'>
<div style='font-size:48px;margin-bottom:16px'>β</div>
<div style='font-family:Sora;font-size:18px;font-weight:700;margin-bottom:8px'>Upload your portfolio CSV to begin</div>
<div style='font-size:12px;color:#64748B;line-height:1.9'>
Required columns: <span style='color:#00FFB2'>Ticker</span>, <span style='color:#00FFB2'>Weight</span>, <span style='color:#00FFB2'>AssetType</span><br>
Supports NSE stocks (RELIANCE.NS), BSE stocks (RELIANCE.BO),<br>ETFs (GOLDBEES.NS), and Mutual Funds by name (HDFC Top 100 Fund)<br><br>
Or click <b>Load Sample</b> to try a pre-built Indian portfolio
</div>
</div>
""", unsafe_allow_html=True)
