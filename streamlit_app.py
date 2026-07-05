import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Hyena Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#1a2e1a; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #4c7a4c; }
.etf-card    { background:linear-gradient(135deg,#1a2e1a 0%,#4c7a4c 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#1a2e1a 0%,#2e4a2e 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🐍 Hyena Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Poli et al. (2023) Hyena Hierarchy · '
    'Implicit MLP-defined long convolution filters, O(L log L) via FFT · '
    'Data-controlled gating — genuinely distinct from S4/Mamba, no state-space recurrence · '
    'Built for very long ETF windows (1008d+)</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## Hyena Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(
    f"**Sequence:** lookback={config.LOOKBACK_LEN} | order={config.HYENA_ORDER} | "
    f"embed dim={config.EMBED_DIM}")
st.sidebar.markdown(
    f"**Filters:** hidden={config.FILTER_HIDDEN} | fourier bands={config.N_FOURIER_FEATURES} | "
    f"decay={config.FILTER_DECAY_RATE}")
st.sidebar.markdown(
    f"**Training:** epochs={config.HYENA_EPOCHS} | lr={config.HYENA_LR} | "
    f"batch={config.HYENA_BATCH_SIZE}")
st.sidebar.markdown(
    f"**Weights:** Forecast {config.WEIGHT_FORECAST:.0%} | "
    f"Long-range {config.WEIGHT_LONGRANGE:.0%} | "
    f"Fit {config.WEIGHT_FIT:.0%}")
st.sidebar.caption(
    "Windows shorter than the lookback + horizon requirement are skipped "
    "automatically — this engine needs long history to be worth using."
)

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        files = [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                           detail=True, recursive=True)
                 if f["type"] == "file"]
        return files, None
    except Exception as e:
        return [], str(e)


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files, list_error = list_repo_files()

with st.expander("🔧 Debug: what the dashboard sees on HuggingFace", expanded=bool(list_error)):
    st.markdown(f"**Repo:** `{OUTPUT_REPO}`  ·  **Token set:** {'yes' if bool(HF_TOKEN) else 'no'}")
    if list_error:
        st.error(f"Could not list repo files: {list_error}")
    else:
        st.write(f"{len(files)} file(s) found:")
        st.code("\n".join(sorted(files)) if files else "(empty)")

tab1_path = find_latest(files, "hyena_engine_2")
tab2_path = find_latest(files, "hyena_engine_windows_")

if not tab1_path:
    if list_error:
        st.error("Could not reach HuggingFace to look for results (see 🔧 Debug above).")
    else:
        st.error(
            "Connected to HuggingFace successfully, but no file matching "
            "`hyena_engine_2*.json` was found (see 🔧 Debug above for the exact "
            "file list). Run trainer.py, or check the filename it actually pushed."
        )
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Long Convolution Forecast Signal")

    with st.expander("Hyena Methodology", expanded=True):
        st.markdown("""
**Two primitives**, sub-quadratic O(L log L) instead of quadratic attention:

**1. Implicit long convolution filters.** Instead of learning L free filter
taps directly, the filter value at position t is the OUTPUT of a small MLP
fed a positional encoding of t, modulated by an exponential decay:

```
h(t) = MLP(pos_features(t)) * exp(-decay_rate * t)
```

The filter is evaluated procedurally for any sequence length — it doesn't
need to be re-learned as L free numbers for each new length.

**2. Data-controlled (gated) recursion.**

```
z^0 = x_1
z^n = x_{n+1} * (h^n conv z^{n-1})     for n = 1..N
y   = z^N
```

`*` is elementwise multiplication — the gate is a function of the input
itself, not fixed weights — and `conv` is a causal long convolution
computed via FFT for O(L log L) complexity, not O(L^2).

**Genuinely distinct from S4/Mamba:** no state-space recurrence anywhere
in this operator, no state transition matrix, no selective/input-dependent
recurrent state. The "memory" here is an explicit convolution against an
implicitly-generated filter, combined with multiplicative gating — a
different computational mechanism entirely.

**Validated before shipping:** the FFT convolution matches a naive direct
O(L^2) computation to floating-point precision, shows empirically growing
speedup as L increases (~3.5x at L=63 up to ~22x at L=1008 — genuine
sub-quadratic scaling, not just a label), and every gradient was checked
against finite differences.

**Signal:**

```
score = 0.50*forecast_signal + 0.25*long_range_utilization + 0.25*fit_quality
```

- `forecast_signal` — predicted mean forward return over the horizon
- `long_range_utilization` — weighted-average lag of the learned implicit
  filters, normalized by the sequence length: is the model actually
  leaning on old information, or behaving like a short-memory model
  despite having a long window available? The one diagnostic here tied
  directly to this specific architectural mechanism.
- `fit_quality` — R² of the trained model on its own training examples

**Built for very long windows.** 63d and 126d are automatically skipped —
this architecture's entire value proposition is efficient use of long
sequences (this engine's lookback is 100+ days per example, and windows
now extend to 1008d/~4 years), so short windows simply aren't long enough
to be worth using it on.
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">Hyena score = {etf['hyena_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
  <div class="etf-score">long-range use = {etf.get('long_range_utilization', float('nan')):.2f}</div>
  <div class="etf-score">fit quality = {etf.get('fit_quality', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    rows.append({
                        "ETF": t,
                        "Hyena Score": info.get("score"),
                        "Best Window (d)": info.get("best_window", "N/A"),
                        "Forecast Signal": info.get("forecast_signal"),
                        "Long-Range Utilization": info.get("long_range_utilization"),
                        "Fit Quality": info.get("fit_quality"),
                    })
                df = pd.DataFrame(rows).sort_values("Hyena Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Poli et al. (2023) Hyena Hierarchy · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore Hyena Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error(
            "No window data available. This engine needs windows substantially "
            "longer than its lookback length — check config.LOOKBACK_LEN vs. "
            "the windows that were actually attempted in the training logs."
        )
        st.stop()

    default_idx  = win_options.index(1008) if 1008 in win_options else len(win_options) - 1
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("Window guidance", expanded=False):
        st.markdown("""
- **252d** — shortest window this engine will typically run on; relatively few training samples
- **504d** — 2-year window; more training samples, more stable filters
- **1008d** — ~4-year window; this is the regime this architecture is actually built for — most training samples, and the most room for the implicit filters to genuinely learn long-range structure
        """)

    st.markdown(f"### Hyena Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">Hyena score = {etf['hyena_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
  <div class="etf-score">long-range use = {etf.get('long_range_utilization', float('nan')):.2f}</div>
  <div class="etf-score">fit quality = {etf.get('fit_quality', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=["ETF", "Hyena Score", "Forecast Signal",
                             "Long-Range Utilization", "Fit Quality"],
                )
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
