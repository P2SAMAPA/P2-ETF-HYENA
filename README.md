# 🐍 P2-ETF-HYENA

**Hyena / Long Convolution Operator Engine — Poli et al. (2023)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine trains a **Hyena** operator per ETF — a sub-quadratic O(L log L)
sequence model built entirely from implicit long convolutions and
data-controlled gating, with **no state-space recurrence at all**. It is
the only engine in this suite specifically designed and tuned for very
long lookback windows (1008d+), where its efficiency advantage over both
quadratic attention and a naive O(L²) direct convolution actually matters.

---

## Theory

### Implicit Long Convolution Filters

Rather than learning `L` free filter taps directly (expensive, and doesn't
generalize across sequence lengths), the filter value at position `t` is
the OUTPUT of a small MLP fed a positional encoding of `t`, modulated by
an exponential decay:

```
h(t) = MLP(pos_features(t)) * exp(-decay_rate * t)
```

The decay modulation keeps the filter numerically well-behaved at long
range — a known requirement from the original paper. The filter is
evaluated procedurally for any sequence length, rather than stored as `L`
learned numbers that would need relearning at a different length.

### Data-Controlled (Gated) Recursion

The operator is defined recursively over an order `N`:

```
z^0 = x_1
z^n = x_{n+1} * (h^n conv z^{n-1})     for n = 1..N
y   = z^N
```

`*` is elementwise multiplication — the gate is a function of the input
itself (data-controlled), not a fixed weight matrix — and `conv` is a
causal long convolution computed via **FFT** for O(L log L) complexity.
`x_1..x_{N+1}` are dense (pointwise) projections of the input, preceded by
a short causal depthwise convolution for local mixing.

### Genuinely Distinct from S4/Mamba

There is no state-space recurrence anywhere in this operator — no state
transition matrix, no selective/input-dependent recurrent state evolving
token by token. The "memory" here is implemented entirely as an explicit
convolution against an implicitly-generated filter, combined with
multiplicative gating — a different computational mechanism, not just a
different parameterization of the same underlying idea.

### Score Construction

```
score = 0.50*forecast_signal + 0.25*long_range_utilization + 0.25*fit_quality
```

| Component | Meaning |
|-----------|---------|
| forecast_signal | Predicted mean forward return over the horizon |
| long_range_utilization | Weighted-average lag of the learned implicit filters, normalized by sequence length — is the model actually leaning on old information? |
| fit_quality | R² of the trained model on its own training examples |

`long_range_utilization` is the one diagnostic in this suite tied directly
to this specific architectural mechanism, rather than a generic fit
measure — it answers whether the model is genuinely exploiting the long
window it was given.

### Validation

Before shipping, the FFT-based causal convolution and its full gradient
chain (embedding → projection → short conv → implicit filters → gating →
output head) were validated:
- **Forward pass** matches a naive direct O(L²) convolution to
  floating-point precision
- **Sub-quadratic scaling confirmed empirically**: ~3.5x speedup over
  direct computation at L=63, growing to ~22x at L=1008
- **Every gradient checked against finite differences.** A genuine bug
  was caught and fixed during this process: an initial per-sample batch
  loop shared mutable layer caches across samples, silently corrupting
  gradients for every sample but the last in each batch. The model was
  rewritten with full batch vectorization throughout, and every parameter
  (weights and biases, across every layer type) now matches finite-
  difference gradients to ~1e-12–1e-14 precision.

---

## Distinction from Other Sequence Models in the Suite

| Engine | Core mechanism | Sequence length regime |
|--------|-----------------|--------------------------|
| Decision Transformer | Causal self-attention | Short context (K=20) |
| N-HiTS | Multi-rate hierarchical pooling | Fixed short lookback |
| **Hyena (this engine)** | **Implicit long convolution + gating** | **Long (100+), built for 1008d+** |

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d · 1008d` — extended beyond the
suite's usual maximum specifically for this engine. 63d and 126d are
automatically skipped (too short relative to `LOOKBACK_LEN`); this is
expected, not a bug — the architecture's entire value proposition is
efficient use of long sequences.

---

## Repository Structure

```
P2-ETF-HYENA/
├── config.py          # Universes, Hyena hyperparameters, score weights
├── data_manager.py    # HuggingFace loader
├── hyena_engine.py      # Core: FFT convolution, implicit filter MLPs, gated recursion
├── trainer.py           # Orchestrator
├── push_results.py      # HfApi.upload_file wrapper
├── streamlit_app.py      # Two-tab Streamlit dashboard
├── us_calendar.py       # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml    # Single job (extended timeout: longer sequences)
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-HYENA
cd P2-ETF-HYENA
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-hyena-results`

---

## References

- Poli, M. et al. (2023). Hyena Hierarchy: Towards Larger Convolutional
  Language Models. ICML 2023.
- Gu, A., Goel, K. & Re, C. (2022). Efficiently Modeling Long Sequences
  with Structured State Spaces (S4). ICLR 2022.
- Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with
  Selective State Spaces.
