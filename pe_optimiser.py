"""
PE Portfolio Optimiser

Run with:  streamlit run pe_optimiser.py

Requirements:  pip install streamlit pandas numpy plotly scipy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
from scipy.stats import skewnorm
from scipy.optimize import minimize

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PE Portfolio Optimiser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #f8f8f7;
        border-radius: 10px;
        padding: 14px 18px;
        border: 0.5px solid #e0dfd8;
    }
    .metric-label { font-size: 11px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 600; color: #111; }
    .metric-sub   { font-size: 11px; color: #999; margin-top: 2px; }
    .insight-box {
        background: #f0f0ee;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #444;
        line-height: 1.65;
        margin-top: 8px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 13px; }
    h1 { font-size: 22px !important; font-weight: 600 !important; }
    h2 { font-size: 17px !important; font-weight: 500 !important; }
    h3 { font-size: 14px !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ── Base data ────────────────────────────────────────────────────────────────

BASE_DATA = {
    "Large Buyout":  {"p5": -3.6,  "p25":  8.6, "p50": 13.9, "p75": 21.0, "p95": 38.0},
    "Mid Buyout":    {"p5": -7.0,  "p25":  7.7, "p50": 13.6, "p75": 21.2, "p95": 37.7},
    "Small Buyout":  {"p5": -15.1, "p25":  7.1, "p50": 15.7, "p75": 23.3, "p95": 41.1},
    "Growth":        {"p5": -7.0,  "p25":  5.6, "p50": 12.5, "p75": 20.7, "p95": 41.7},
    "Late VC":       {"p5": -11.3, "p25":  4.2, "p50": 12.1, "p75": 20.4, "p95": 40.6},
    "Early VC":      {"p5": -13.6, "p25":  2.0, "p50": 10.1, "p75": 20.3, "p95": 38.9},
}

RETURN_PRESETS = {
    "Historic": {s: BASE_DATA[s]["p50"] for s in BASE_DATA},
    "Forecast": {
        "Large Buyout":  8.6,
        "Mid Buyout":    9.1,
        "Small Buyout": 10.1,
        "Growth":        9.6,
        "Late VC":      10.0,
        "Early VC":     12.0,
    },
}

STRATEGY_COLORS = {
    "Large Buyout": "#185FA5",
    "Mid Buyout":   "#0C447C",
    "Small Buyout": "#378ADD",
    "Growth":       "#0F6E56",
    "Late VC":      "#993C1D",
    "Early VC":     "#712B13",
}

BOUNDS = {
    "Large Buyout": (10, 35),
    "Mid Buyout":   (10, 35),
    "Small Buyout": ( 5, 25),
    "Growth":       (10, 30),
    "Late VC":      ( 5, 20),
    "Early VC":     ( 5, 20),
}

# Illustrative cross-strategy return correlation matrix.
# Rows/columns follow the order of BASE_DATA keys.
# Buyout sub-strategies are highly correlated (~0.75–0.85); VC sub-strategies
# are highly correlated with each other (~0.75) but weakly with Buyout (~0.25–0.45).
STRATEGIES = list(BASE_DATA.keys())  # index reference for CORRELATION

CORRELATION = np.array([
    #            LBO   MBO   SBO   GRW   LVC   EVC
    [1.00, 0.85, 0.75, 0.50, 0.35, 0.25],  # Large Buyout
    [0.85, 1.00, 0.80, 0.55, 0.40, 0.30],  # Mid Buyout
    [0.75, 0.80, 1.00, 0.60, 0.45, 0.35],  # Small Buyout
    [0.50, 0.55, 0.60, 1.00, 0.65, 0.55],  # Growth
    [0.35, 0.40, 0.45, 0.65, 1.00, 0.75],  # Late VC
    [0.25, 0.30, 0.35, 0.55, 0.75, 1.00],  # Early VC
])

# ── Helper functions ──────────────────────────────────────────────────────────

def compute_stats(data):
    out = {}
    for strat, d in data.items():
        iqr    = d["p75"] - d["p25"]
        up_gap = d["p95"] - d["p50"]
        dn_gap = d["p50"] - d["p5"]
        skew   = "High" if up_gap > dn_gap * 1.8 else "Moderate" if up_gap > dn_gap * 1.3 else "Low"
        out[strat] = {**d, "iqr": iqr, "skew": skew}
    return out

def generate_grid(step=5):
    strategies = list(BOUNDS.keys())
    n = len(strategies)
    results = []

    def recurse(idx, remaining, current):
        if idx == n - 1:
            lo, hi = BOUNDS[strategies[idx]]
            if lo <= remaining <= hi:
                results.append(current + [remaining])
            return
        lo, hi = BOUNDS[strategies[idx]]
        min_rest = sum(BOUNDS[strategies[j]][0] for j in range(idx + 1, n))
        for v in range(lo, min(hi, remaining - min_rest) + 1, step):
            recurse(idx + 1, remaining - v, current + [v])

    recurse(0, 100, [])
    return results

def fit_skewnorm_to_percentiles(p5, p25, p50, p75, p95):
    """Fit a skew-normal distribution to 5 percentile points via least-squares."""
    known_q = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    known_v = np.array([p5, p25, p50, p75, p95])

    def residuals(params):
        a, loc, scale = params
        if scale <= 0:
            return 1e10
        fitted = skewnorm.ppf(known_q, a, loc=loc, scale=scale)
        return np.sum((fitted - known_v) ** 2)

    loc0   = p50
    scale0 = (p75 - p25) / 1.349
    result = minimize(residuals, [0.0, loc0, scale0], method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 10000})
    return result.x  # a, loc, scale


@st.cache_data
def precompute_grid():
    return generate_grid(step=5)

def _corr_adjusted_iqr(wf, stats, strategies):
    """Portfolio IQR using the correlation matrix (IQR ≈ 1.349 × σ for normal)."""
    corr = st.session_state.get("corr_matrix", CORRELATION)
    sigmas = np.array([stats[s]["iqr"] / 1.349 for s in strategies])
    w = np.array(wf)
    port_var = float(w @ np.diag(sigmas) @ corr @ np.diag(sigmas) @ w)
    return 1.349 * np.sqrt(max(port_var, 0.0))


def _portfolio_metrics(weights, stats, mgr_prob):
    strategies = list(stats.keys())
    wf = [w / 100 for w in weights]

    irr   = sum(wf[i] * stats[strategies[i]]["p25"]  for i in range(len(wf)))
    med   = sum(wf[i] * stats[strategies[i]]["p50"]  for i in range(len(wf)))
    p75   = sum(wf[i] * stats[strategies[i]]["p75"]  for i in range(len(wf)))
    cvar  = sum(wf[i] * stats[strategies[i]]["p5"]   for i in range(len(wf)))
    iqr   = _corr_adjusted_iqr(wf, stats, strategies)

    scen = sum(
        wf[i] * (
            mgr_prob * stats[strategies[i]]["p25"]
            + (1 - mgr_prob) * 0.7 * stats[strategies[i]]["p50"]
            + (1 - mgr_prob) * 0.3 * (stats[strategies[i]]["p25"] - stats[strategies[i]]["iqr"])
        )
        for i in range(len(wf))
    )
    return {"irr": irr, "med": med, "p75": p75, "iqr": iqr, "cvar": cvar, "scen": scen}


def score_portfolio(weights, stats, method, param, mgr_prob):
    metrics = _portfolio_metrics(weights, stats, mgr_prob)

    if method == "mvd":
        sc = metrics["irr"] - param * metrics["iqr"]
    elif method == "cvar":
        if metrics["cvar"] < -param:
            return None
        sc = metrics["irr"]
    else:
        sc = metrics["scen"]

    return {"sc": sc, **metrics}


def _make_feasible_start(strategies, bounds_list, preference=None):
    lo = np.array([b[0] for b in bounds_list], dtype=float)
    hi = np.array([b[1] for b in bounds_list], dtype=float)
    cap = hi - lo
    remaining = 100.0 - float(lo.sum())
    if remaining < 0:
        raise ValueError("Lower-bound sum exceeds 100%.")

    if preference is None:
        pref = np.ones(len(strategies), dtype=float)
    else:
        pref = np.maximum(np.array(preference, dtype=float), 0.0)

    alloc_weights = pref * cap
    if alloc_weights.sum() <= 0:
        alloc_weights = cap.copy()

    add = remaining * alloc_weights / alloc_weights.sum()
    return lo + add

def optimise(stats, method, param, mgr_prob, apply_bounds=True):
    strategies = list(stats.keys())
    bounds_list = [BOUNDS[s] for s in strategies] if apply_bounds else [(0, 100)] * len(strategies)

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 100.0)}]
    if method == "cvar":
        constraints.append({
            "type": "ineq",
            "fun": lambda w: float(_portfolio_metrics(w, stats, mgr_prob)["cvar"] + param),
        })

    starts = [_make_feasible_start(strategies, bounds_list)]
    for i in range(len(strategies)):
        pref = np.zeros(len(strategies))
        pref[i] = 1.0
        starts.append(_make_feasible_start(strategies, bounds_list, preference=pref))

    rng = np.random.default_rng(42)
    for _ in range(18):
        starts.append(_make_feasible_start(strategies, bounds_list, preference=rng.random(len(strategies))))

    best, best_sc = None, -np.inf
    for x0 in starts:
        def objective(w):
            result = score_portfolio(w, stats, method, param, mgr_prob)
            return -result["sc"] if result is not None else 1e9

        opt = minimize(
            fun=objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds_list,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if not opt.success:
            continue

        result = score_portfolio(opt.x, stats, method, param, mgr_prob)
        if result and result["sc"] > best_sc:
            best_sc = result["sc"]
            best = {
                "weights": [float(x) for x in opt.x],
                "strategies": strategies,
                **result,
            }

    return best

def build_frontier(stats, mgr_prob, apply_bounds=True):
    strategies = list(stats.keys())

    # Scatter: bounded grid for visualising the feasibility region
    grid = precompute_grid()
    scatter_rows = []
    for w in grid:
        wf = [x / 100 for x in w]
        irr = sum(wf[i] * stats[strategies[i]]["p25"] for i in range(len(wf)))
        iqr = _corr_adjusted_iqr(wf, stats, strategies)
        scatter_rows.append({"irr": round(irr, 3), "iqr": round(iqr, 3)})
    df_all = pd.DataFrame(scatter_rows)

    # Efficient frontier: sweep lambda to trace the upper envelope via continuous optimiser.
    # Low lambda → high IRR, high IQR (upper-right).
    # High lambda → low IQR, low IRR (lower-left).
    # Together they give an upward-sloping IRR-vs-IQR frontier.
    lambdas = np.concatenate([
        np.linspace(0.0, 0.4, 20),
        np.linspace(0.4, 2.0, 15),
        np.linspace(2.0, 10.0, 10),
    ])
    frontier_rows = []
    seen: set = set()
    for lam in lambdas:
        r = optimise(stats, "mvd", float(lam), mgr_prob, apply_bounds=apply_bounds)
        if r:
            key = (round(r["irr"], 2), round(r["iqr"], 2))
            if key not in seen:
                seen.add(key)
                frontier_rows.append({"irr": r["irr"], "iqr": r["iqr"]})

    df_front = (pd.DataFrame(frontier_rows).sort_values("iqr").reset_index(drop=True)
                if frontier_rows else pd.DataFrame(columns=["irr", "iqr"]))
    return df_all, df_front

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Portfolio settings")

    st.markdown("### Optimisation method")
    method = st.radio(
        "Select method",
        options=["mvd", "cvar", "scenario"],
        format_func=lambda x: {
            "mvd":      "A — IQR-penalised (mean-variance)",
            "cvar":     "B — P5 floor (tail-loss limit)",
            "scenario": "C — Scenario-weighted",
        }[x],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if method == "mvd":
        st.markdown("### Dispersion aversion (λ)")
        lam = st.slider("λ — penalty per unit of IQR", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        param = lam
        st.caption("Higher λ → more weight to low-dispersion strategies (Buyout).")
    elif method == "cvar":
        st.markdown("### P5 floor")
        cvar_floor = st.slider("Maximum acceptable tail loss (%)", min_value=3, max_value=20, value=10, step=1)
        param = cvar_floor
        st.caption("Tighten to exclude high-tail-risk strategies (Small Buyout, Early VC).")
    else:
        st.markdown("### Manager quality")
        mgr_pct = st.slider("Top-quartile manager probability (%)", min_value=10, max_value=90, value=60, step=5)
        param = mgr_pct / 100
        st.caption("25% = new LP. 60–70% = established LP with strong GP relationships.")

    if method != "scenario":
        mgr_pct = 60
        param_mgr = 0.60
    else:
        param_mgr = param

    st.markdown("---")
    st.markdown("### Return expectations (median IRR)")
    st.caption("Enter expected median IRR per strategy. IQR is preserved — all percentiles shift with the median.")

    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("Load Historic", use_container_width=True):
            for s, v in RETURN_PRESETS["Historic"].items():
                st.session_state[f"ret_{s}"] = v
            st.rerun()
    with preset_col2:
        if st.button("Load Forecast", use_container_width=True):
            for s, v in RETURN_PRESETS["Forecast"].items():
                st.session_state[f"ret_{s}"] = v
            st.rerun()

    strategies = list(BASE_DATA.keys())
    median_adjustments = {}
    for strat in strategies:
        base_med = BASE_DATA[strat]["p50"]
        expected = st.number_input(
            strat, min_value=-20.0, max_value=60.0, value=base_med, step=0.1,
            format="%.1f", key=f"ret_{strat}",
            help=f"Historical base: {base_med:.1f}%",
        )
        median_adjustments[strat] = round(expected - base_med, 4)


# ── Build adjusted data ───────────────────────────────────────────────────────

adjusted_data = {}
for strat, base in BASE_DATA.items():
    delta = median_adjustments[strat]
    adjusted_data[strat] = {k: round(v + delta, 2) for k, v in base.items()}

stats = compute_stats(adjusted_data)

# ── Main layout ───────────────────────────────────────────────────────────────

st.markdown("# PE Portfolio Optimiser")

tab_main, tab_data, tab_frontier, tab_dist, tab_method = st.tabs([
    "📐 Optimisation", "📋 Strategy data", "📈 Efficient frontier", "🔔 Distributions", "📖 Methodology"
])

with tab_main:
    ctrl_left, ctrl_mid, _ = st.columns([1, 2, 7])
    with ctrl_left:
        if st.button("🔄 Refresh", key="refresh_optimisation"):
            st.cache_data.clear()
            st.rerun()
    with ctrl_mid:
        apply_bounds = st.toggle("Apply allocation bounds", value=True, key="apply_bounds")
    res = optimise(stats, method, param if method != "scenario" else 0, param_mgr, apply_bounds=apply_bounds)

    if res is None:
        st.error("No feasible allocation found at this constraint level. Try loosening the floor.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio IRR (P25-weighted)</div>
            <div class="metric-value" style="color:#185FA5;">{res['irr']:.1f}%</div>
            <div class="metric-sub">Bottom-quartile return floor</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        if method == "mvd":
            val  = f"{res['irr'] - (param * res['iqr']):.1f}%"
            lbl  = "Penalty-adjusted return"
            sub  = f"IRR − {param:.2f}×IQR"
            col  = "#185FA5"
        elif method == "cvar":
            val  = f"{res['cvar']:.1f}%"
            lbl  = "Portfolio P5 (tail quantile)"
            sub  = "5th-percentile outcome"
            col  = "#A32D2D" if res['cvar'] < -10 else "#1a7f37"
        else:
            val  = f"{res['scen']:.1f}%"
            lbl  = "Scenario-weighted IRR"
            sub  = f"{int(param_mgr*100)}/{ int((1-param_mgr)*70) }/{ int((1-param_mgr)*30) } prob. mix"
            col  = "#185FA5"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:{col};">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        disp_col = "#1a7f37" if res['iqr'] < 14 else "#856404" if res['iqr'] < 16 else "#A32D2D"
        disp_lbl = "Low" if res['iqr'] < 14 else "Moderate" if res['iqr'] < 16 else "High"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Dispersion score (IQR)</div>
            <div class="metric-value" style="color:{disp_col};">{res['iqr']:.1f}pp</div>
            <div class="metric-sub">{disp_lbl} dispersion</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        vc_w  = res["weights"][4] + res["weights"][5]
        bo_w  = res["weights"][0] + res["weights"][1] + res["weights"][2]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Strategy mix</div>
            <div class="metric-value">{bo_w:.1f}% / {res['weights'][3]:.1f}% / {vc_w:.1f}%</div>
            <div class="metric-sub">Buyout / Growth / VC</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Optimal allocation")
        rows = []
        for i, strat in enumerate(res["strategies"]):
            w = res["weights"][i]
            s = stats[strat]
            rows.append({"Strategy": strat, "Weight": f"{w:.1f}%",
                         "P25 IRR": f"{s['p25']:.1f}%", "P50 Median": f"{s['p50']:.1f}%", "P75 IRR": f"{s['p75']:.1f}%",
                         "IQR": f"{s['iqr']:.1f}pp"})
        rows.append({"Strategy": "**Portfolio**", "Weight": "**100%**",
                     "P25 IRR": f"**{res['irr']:.1f}%**", "P50 Median": f"**{res['med']:.1f}%**", "P75 IRR": f"**{res['p75']:.1f}%**",
                     "IQR": f"**{res['iqr']:.1f}pp**"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col_right:
        st.markdown("#### Allocation mix")
        fig_bar = go.Figure()
        for i, strat in enumerate(res["strategies"]):
            fig_bar.add_trace(go.Bar(
                name=strat, x=[res["weights"][i]], y=["Portfolio"], orientation="h",
                marker_color=STRATEGY_COLORS[strat],
                text=f"{strat}<br>{res['weights'][i]:.1f}%", textposition="inside", insidetextanchor="middle",
                hovertemplate=f"<b>{strat}</b><br>Weight: {res['weights'][i]:.1f}%<extra></extra>",
            ))
        fig_bar.update_layout(barmode="stack", height=120, margin=dict(l=0, r=0, t=10, b=10), showlegend=False,
            xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False), yaxis=dict(showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = go.Figure(go.Pie(
            labels=res["strategies"], values=res["weights"], hole=0.55,
            marker_colors=[STRATEGY_COLORS[s] for s in res["strategies"]],
            textinfo="label+percent", textfont_size=11,
            hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>",
        ))
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    if not apply_bounds:
        st.info("Allocation bounds disabled — optimizer is unconstrained (weights sum to 100% only).")
    elif st.toggle("Show allocation bounds", value=False, key="show_bounds"):
        bounds_rows = [{"Strategy": s, "Min weight": f"{lo}%", "Max weight": f"{hi}%"}
                       for s, (lo, hi) in BOUNDS.items()]
        st.dataframe(pd.DataFrame(bounds_rows), hide_index=True, use_container_width=True)

    if method == "mvd":
        early_adj = stats["Early VC"]["p25"] - param * stats["Early VC"]["iqr"]
        lb_adj    = stats["Large Buyout"]["p25"] - param * stats["Large Buyout"]["iqr"]
        insight = (f"At λ={param:.2f}, the IQR penalty reduces Early VC's effective return to {early_adj:.1f}% "
                   f"(IQR {stats['Early VC']['iqr']:.1f}pp) and Large Buyout's to {lb_adj:.1f}% "
                   f"(IQR {stats['Large Buyout']['iqr']:.1f}pp). Portfolio IQR after correlation adjustment: "
                   f"{res['iqr']:.1f}pp. Buyout dominates at {bo_w:.1f}% and VC at {vc_w:.1f}%.")
    elif method == "cvar":
        insight = (f"P5 floor set at −{param:.0f}%. Small Buyout (P5=−{stats['Small Buyout']['p5']:.1f}%) and Early VC (P5=−{stats['Early VC']['p5']:.1f}%) are the binding tail-risk strategies. Portfolio P5 = {res['cvar']:.1f}%.")
    else:
        evc_scen = (param_mgr * stats["Early VC"]["p25"] + (1 - param_mgr) * 0.7 * stats["Early VC"]["p50"] + (1 - param_mgr) * 0.3 * (stats["Early VC"]["p25"] - stats["Early VC"]["iqr"]))
        lb_scen  = (param_mgr * stats["Large Buyout"]["p25"] + (1 - param_mgr) * 0.7 * stats["Large Buyout"]["p50"] + (1 - param_mgr) * 0.3 * (stats["Large Buyout"]["p25"] - stats["Large Buyout"]["iqr"]))
        insight = (f"At {int(param_mgr*100)}% top-quartile probability: Early VC scenario-weighted IRR = {evc_scen:.1f}% vs Large Buyout {lb_scen:.1f}%.")

    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

with tab_data:
    st.markdown("#### Strategy return & dispersion profiles")
    rows = []
    for strat, s in stats.items():
        delta = median_adjustments[strat]
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%" if delta < 0 else "—"
        skew_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(s["skew"], "")
        rows.append({"Strategy": strat, "P5 (tail quantile)": f"{s['p5']:.1f}%", "P25 (Q1 IRR)": f"{s['p25']:.1f}%",
            "P50 (Median)": f"{s['p50']:.1f}%", "P75 (Q3)": f"{s['p75']:.1f}%", "P95": f"{s['p95']:.1f}%",
            "IQR": f"{s['iqr']:.1f}pp", "Median adj.": delta_str, "Skew": f"{skew_emoji} {s['skew']}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Cross-strategy return correlation")
    st.caption("Edit values directly. Matrix is symmetrised and diagonal fixed at 1.0 on each change. Used in portfolio IQR calculation.")

    if "corr_matrix" not in st.session_state:
        st.session_state["corr_matrix"] = CORRELATION.copy()

    short_names = ["Lg Buyout", "Mid Buyout", "Sm Buyout", "Growth", "Late VC", "Early VC"]
    corr_df = pd.DataFrame(st.session_state["corr_matrix"], index=short_names, columns=short_names)

    edited_corr_df = st.data_editor(
        corr_df, use_container_width=True, key="corr_editor",
        column_config={col: st.column_config.NumberColumn(col, min_value=-0.99, max_value=0.99, step=0.05, format="%.2f")
                       for col in short_names},
    )

    new_corr = edited_corr_df.values.astype(float)
    new_corr = np.clip(new_corr, -0.99, 0.99)
    new_corr = (new_corr + new_corr.T) / 2
    np.fill_diagonal(new_corr, 1.0)
    if not np.allclose(new_corr, st.session_state["corr_matrix"]):
        st.session_state["corr_matrix"] = new_corr
        st.rerun()

    fig_corr = go.Figure(go.Heatmap(
        z=new_corr, x=short_names, y=short_names,
        colorscale="Blues", zmin=0, zmax=1,
        text=[[f"{new_corr[i][j]:.2f}" for j in range(6)] for i in range(6)],
        texttemplate="%{text}", textfont_size=11,
        hovertemplate="<b>%{y} / %{x}</b><br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig_corr.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    fig_fan = go.Figure()
    for strat in list(stats.keys()):
        s = stats[strat]
        col = STRATEGY_COLORS[strat]
        fig_fan.add_trace(go.Scatter(x=[strat, strat], y=[s["p5"], s["p95"]], mode="lines", line=dict(color=col, width=12), opacity=0.18, showlegend=False, hoverinfo="skip"))
        fig_fan.add_trace(go.Scatter(x=[strat, strat], y=[s["p25"], s["p75"]], mode="lines", line=dict(color=col, width=12), opacity=0.55, showlegend=False, hoverinfo="skip"))
        fig_fan.add_trace(go.Scatter(x=[strat], y=[s["p50"]], mode="markers", marker=dict(color=col, size=12, line=dict(color="white", width=2)), name=strat,
            hovertemplate=f"<b>{strat}</b><br>P5: {s['p5']:.1f}%<br>P25: {s['p25']:.1f}%<br>Median: {s['p50']:.1f}%<br>P75: {s['p75']:.1f}%<br>P95: {s['p95']:.1f}%<br>IQR: {s['iqr']:.1f}pp<extra></extra>"))
    fig_fan.update_layout(height=420, yaxis=dict(title="Net IRR (%)", ticksuffix="%", zeroline=True, zerolinecolor="#ddd"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(l=40, r=20, t=40, b=20))
    st.plotly_chart(fig_fan, use_container_width=True)
    st.caption("Thick band = IQR (P25–P75). Thin band = P5–P95. Dot = median.")

with tab_frontier:
    st.markdown("#### Efficient frontier — expected IRR vs dispersion (IQR)")
    with st.spinner("Computing frontier…"):
        df_all, df_front = build_frontier(stats, param_mgr, apply_bounds=apply_bounds)

    opt_iqr = round(res["iqr"], 2)
    opt_irr = round(res["irr"], 2)

    fig_front = go.Figure()
    fig_front.add_trace(go.Scatter(x=df_all["iqr"], y=df_all["irr"], mode="markers", marker=dict(color="#ccc", size=4, opacity=0.4), name="All feasible portfolios"))
    df_front_sorted = df_front.sort_values("iqr")
    fig_front.add_trace(go.Scatter(x=df_front_sorted["iqr"], y=df_front_sorted["irr"], mode="lines+markers", line=dict(color="#185FA5", width=2.5), name="Efficient frontier"))
    fig_front.add_trace(go.Scatter(x=[opt_iqr], y=[opt_irr], mode="markers", marker=dict(symbol="star", color="#993C1D", size=20, line=dict(color="white", width=1.5)), name="Current optimal"))
    fig_front.update_layout(height=440, xaxis=dict(title="Portfolio dispersion score (IQR, pp)", ticksuffix="pp"), yaxis=dict(title="Expected IRR (%)", ticksuffix="%"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_front, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Method comparison")
    comp_rows = []
    for label, m, p in [
        ("A — IQR penalty λ=0.25", "mvd", 0.25), ("A — IQR penalty λ=0.50", "mvd", 0.50), ("A — IQR penalty λ=0.75", "mvd", 0.75),
        ("B — P5 floor −5%", "cvar", 5), ("B — P5 floor −10%", "cvar", 10), ("B — P5 floor −15%", "cvar", 15),
        ("C — Scenario 40% top-Q", "scenario", 0.40), ("C — Scenario 60% top-Q", "scenario", 0.60), ("C — Scenario 80% top-Q", "scenario", 0.80),
    ]:
        mgr = p if m == "scenario" else 0.60
        r = optimise(stats, m, p if m != "scenario" else 0, mgr)
        if r:
            w = r["weights"]
            comp_rows.append({"Method / param": label, "Large BO": f"{w[0]:.1f}%", "Mid BO": f"{w[1]:.1f}%", "Small BO": f"{w[2]:.1f}%",
                "Growth": f"{w[3]:.1f}%", "Late VC": f"{w[4]:.1f}%", "Early VC": f"{w[5]:.1f}%", "IRR": f"{r['irr']:.1f}%", "IQR": f"{r['iqr']:.1f}pp", "P5": f"{r['cvar']:.1f}%"})
        else:
            comp_rows.append({"Method / param": label, "Large BO": "—", "Mid BO": "—", "Small BO": "—", "Growth": "—", "Late VC": "—", "Early VC": "—", "IRR": "—", "IQR": "—", "P5": "Infeasible"})
    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

with tab_dist:
    st.markdown("#### Strategy return distributions")
    selected = st.multiselect("Strategies to display", options=list(stats.keys()), default=["Large Buyout", "Growth", "Early VC"])

    if selected:
        fig_dist = go.Figure()
        x_range = np.linspace(-35, 65, 500)
        for strat in selected:
            s = stats[strat]
            col = STRATEGY_COLORS[strat]
            a, loc, scale = fit_skewnorm_to_percentiles(s["p5"], s["p25"], s["p50"], s["p75"], s["p95"])
            pdf = skewnorm.pdf(x_range, a, loc=loc, scale=scale)
            if pdf.max() > 0:
                pdf /= pdf.max()
            fig_dist.add_trace(go.Scatter(x=x_range, y=pdf, mode="lines", name=strat, line=dict(color=col, width=2.5)))
            mask = (x_range >= s["p25"]) & (x_range <= s["p75"])
            fig_dist.add_trace(go.Scatter(x=np.concatenate([x_range[mask], x_range[mask][::-1]]), y=np.concatenate([pdf[mask], np.zeros(mask.sum())]),
                fill="toself", fillcolor=col, opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            med_pdf = float(np.interp(s["p50"], x_range, pdf))
            fig_dist.add_trace(go.Scatter(x=[s["p50"], s["p50"]], y=[0, med_pdf], mode="lines", line=dict(color=col, width=1.5), showlegend=False, hoverinfo="skip"))
        fig_dist.update_layout(height=420, xaxis=dict(title="Net IRR (%)", ticksuffix="%", zeroline=True, zerolinecolor="#ddd"),
            yaxis=dict(title="Relative frequency", showticklabels=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(l=20, r=20, t=40, b=40))
        st.plotly_chart(fig_dist, use_container_width=True)

        rows = []
        for strat in selected:
            s = stats[strat]
            rows.append({"Strategy": strat, "P5": f"{s['p5']:.1f}%", "P25": f"{s['p25']:.1f}%", "P50": f"{s['p50']:.1f}%",
                "P75": f"{s['p75']:.1f}%", "P95": f"{s['p95']:.1f}%", "IQR": f"{s['iqr']:.1f}pp",
                "Upside (P95−P50)": f"{s['p95']-s['p50']:.1f}pp", "Downside (P50−P5)": f"{s['p50']-s['p5']:.1f}pp"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with tab_method:
    st.markdown("#### Methodology")
    st.markdown("""
**Data inputs**

The tool uses five percentile anchors per strategy — P5, P25, P50, P75, and P95 — representing the distribution of
net IRR outcomes (USD) across a vintage-year universe of funds. These are illustrative distributions calibrated to
observed private markets return data. The median adjustment sliders shift all five percentiles equally, preserving
the IQR and distribution shape while allowing the user to express views on future performance relative to history.

---

**Distribution fitting**

Return distributions are fitted using a **skew-normal distribution** parameterised by location (ξ), scale (ω), and
shape (α). Given five percentile constraints, the three parameters are estimated by minimising the sum of squared
errors between the fitted distribution's theoretical quantiles and the observed anchors (Nelder-Mead optimisation).
This produces smooth, continuous distributions with realistic positive skew for buyout strategies and heavier left
tails for venture, consistent with the asymmetric payoff profiles observed empirically in private markets.

Formally, a random variable X follows a skew-normal distribution if its PDF is:

> f(x) = (2/ω) · φ((x−ξ)/ω) · Φ(α·(x−ξ)/ω)

where φ and Φ are the standard normal PDF and CDF respectively.

---

**Portfolio construction**

Portfolios are constructed as linear combinations of strategy weights subject to per-strategy allocation bounds:

| Strategy | Min | Max |
|---|---|---|
| Large Buyout | 10% | 35% |
| Mid Buyout | 10% | 35% |
| Small Buyout | 5% | 25% |
| Growth | 10% | 30% |
| Late VC | 5% | 20% |
| Early VC | 5% | 20% |

Portfolio optimisation is solved in continuous weight space using constrained nonlinear programming
(SLSQP), with per-strategy bounds and a full-investment constraint (weights sum to 100%). Portfolio expected
return percentiles (P25, P50, P75, P5) are computed as weighted averages of strategy-level percentiles.
Portfolio dispersion (IQR) is computed using the correlation-adjusted variance formula described below.

---

**Cross-strategy correlation and portfolio IQR**

Each strategy's IQR is converted to an approximate standard deviation via σᵢ = IQRᵢ / 1.349 (exact for a normal
distribution; a reasonable approximation for the mildly skewed PE return distributions used here).

Portfolio variance is then:

> σ²_p = wᵀ · Σ · w = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ

where ρ is the cross-strategy correlation matrix (shown on the Strategy data tab) and w is the vector of portfolio
weights. Portfolio IQR is recovered as:

> IQRₚ = 1.349 × √σ²_p

This formulation captures the diversification benefit of combining low-correlation strategies. A portfolio mixing
Buyout and VC will show lower portfolio IQR than the weighted average of the individual strategy IQRs,
consistent with observed PE portfolio outcomes.

The illustrative correlation assumptions are:
- **Buyout–Buyout**: 0.75–0.85 (same vintage cycle exposure, similar deal sourcing)
- **VC–VC**: 0.75 (shared technology sector and exit market sensitivity)
- **Buyout–VC**: 0.25–0.55 (meaningfully lower; different value-creation mechanisms and exit timings)
- **Growth**: intermediate at 0.50–0.65 (bridges Buyout and VC dynamics)

---

**Optimisation methods**

**Method A — IQR-penalised return (Mean-Variance Dispersion)**

Maximises a penalty-adjusted score:

> score = IRR − λ × IQRₚ

where IRR is the portfolio P25-weighted return, IQRₚ is the correlation-adjusted portfolio IQR, and λ is the
dispersion aversion parameter. Higher λ shifts the optimal portfolio toward lower-dispersion strategies (Buyout).
At λ = 0 the method reduces to pure IRR maximisation.

**Method B — P5-constrained (tail quantile floor)**

Maximises portfolio P25 IRR subject to a tail-risk floor: the portfolio P5 return must exceed −floor%. This
excludes allocations where the worst 5% outcome falls below the user-specified threshold, effectively capping
exposure to strategies with severe left-tail risk (Small Buyout, Early VC).

**Method C — Scenario-weighted**

Blends two manager outcome scenarios weighted by the user's assumed probability of top-quartile GP access (p):

> score = p × P25 + (1−p) × [0.7 × P50 + 0.3 × (P25 − IQR)]

The first term represents the return achievable with top-quartile managers; the second is a weighted average of
a median-manager outcome (70% weight) and a stress scenario (30% weight). Lower p values reflect a newer LP with
less established GP relationships.

---

**Limitations**

- The correlation matrix is illustrative. Actual correlations vary by vintage year and are estimated with significant
  uncertainty from the limited history of PE return data.
- IQR-to-σ conversion assumes normality (factor 1.349). The skew-normal distributions fitted here have somewhat
  different IQR/σ ratios; this introduces a small approximation error in the portfolio IQR.
- Portfolio P5 remains a weighted average and does not benefit from the correlation adjustment, which
  underestimates tail-risk diversification.
- The optimisation is continuous, but the efficient-frontier scatter currently uses a 5% grid for visualisation.
- All return assumptions are illustrative and should be replaced with manager-specific data for live use.
- The model does not account for J-curve dynamics, liquidity constraints, or secondary market valuations.
""")

st.markdown("---")
st.caption("Net IRR (USD) · All returns are illustrative distributions. Not investment advice.")
