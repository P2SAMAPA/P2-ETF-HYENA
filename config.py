import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-hyena-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "SMH", "SOXX", "XLB",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "SMH", "SOXX", "XLB",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
# Extended with 1008d (~4 years) beyond the suite's usual max of 504d: this
# engine's entire value proposition is efficient use of VERY long sequences,
# so windows short relative to LOOKBACK_LEN are naturally skipped (see
# trainer.py) rather than padded/faked — 63d and 126d will typically produce
# no result for this engine, which is expected and correctly reflects that
# the architecture needs long history to be worth using at all.
WINDOWS = [63, 126, 252, 504, 1008]

# ── Hyena hyperparameters ─────────────────────────────────────────────────────
# Poli et al. (2023) "Hyena Hierarchy: Towards Larger Convolutional Language
# Models". A sub-quadratic O(L log L) alternative to both quadratic
# attention and structured state-space models (S4/Mamba). Two primitives:
#
#   1. IMPLICIT LONG CONVOLUTION FILTERS — instead of learning L free filter
#      taps directly (expensive, doesn't generalize across sequence
#      lengths), the filter h(t) at each position t is the OUTPUT of a small
#      MLP fed a positional encoding of t, modulated by an exponential decay
#      window. The filter is evaluated procedurally for any L.
#
#   2. DATA-CONTROLLED (GATED) RECURSION — the operator is defined
#      recursively over an order N:
#          z^0 = x_1
#          z^n = x_{n+1} * (h^n conv z^{n-1})    (n = 1..N)
#          y   = z^N
#      where "*" is ELEMENTWISE multiplication (data-controlled gating —
#      the gate is a function of the input itself, not fixed weights) and
#      "conv" is a CAUSAL long convolution, computed via FFT for O(L log L)
#      complexity rather than O(L^2).
#
# This is a GENUINELY DISTINCT mechanism from S4/Mamba: no state-space
# recurrence at all, no selective/input-dependent state transition matrix —
# just explicit convolution (with an implicitly-parameterized filter) plus
# multiplicative gating between dense projections of the input.

LOOKBACK_LEN = 100     # L: sequence length fed into the Hyena operator per
                        # training example — genuinely long relative to the
                        # short fixed contexts used elsewhere in this suite
                        # (e.g. the Decision Transformer's CONTEXT_LEN=20)
HYENA_ORDER   = 2        # N: recursive gating depth (paper's common default)
EMBED_DIM     = 8        # D: channel dimension after the input projection
FILTER_HIDDEN = 16       # hidden width of each implicit filter MLP
N_FOURIER_FEATURES = 5   # sinusoidal positional feature bands fed to the filter MLP
FILTER_DECAY_RATE   = 0.02   # exponential decay modulation applied to each implicit filter
SHORT_CONV_WIDTH    = 3      # local short conv kernel width (local mixing before the long-conv recursion)

PRED_HORIZON = 21        # H: forward return horizon defining the regression target

HYENA_EPOCHS     = 60
HYENA_LR         = 3e-3
HYENA_BATCH_SIZE = 16

# ── Score construction ────────────────────────────────────────────────────────
# forecast_signal        : the model's predicted mean forward return
# long_range_utilization : weighted-average lag (in units of the window
#                          length) of the learned implicit filters — how much
#                          the model is actually leaning on OLD information
#                          vs. behaving like a short-memory model despite
#                          having a long window available. This is the one
#                          diagnostic in this suite tied directly to the
#                          specific architectural mechanism (implicit long
#                          convolution) rather than a generic fit measure.
# fit_quality            : R^2 of the trained model on its own training
#                          examples for this ticker/window

WEIGHT_FORECAST   = 0.50
WEIGHT_LONGRANGE   = 0.25
WEIGHT_FIT          = 0.25

TOP_N = 3
