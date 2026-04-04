"""
models.py — Fair value and macro models.

  YieldModel   — derives current yield y_t from Fed prediction markets and CPI news.
  StockAModel  — FV_A = EPS_A × PE_t  (earnings + rate-sensitive P/E)
  StockCModel  — FV_C = EPS_C × PE_t + lambda × (ΔB / N)  (earnings + bond portfolio)

Phase 5 will add:
  ETFModel     — FV_ETF = FV_A + FV_B + FV_C
"""

import math
from typing import Optional
from state import MarketState


# ---------------------------------------------------------------------------
# Parameters — tune these against historical data before going live.
# ---------------------------------------------------------------------------

# Baseline yield. Set to the prevailing risk-free rate at competition start.
# All yield-sensitive models use (y_t - Y_0) as their input, so this acts
# as an anchor: if the market implies no rate change, y_t == Y_0.
Y_0: float = 0.045        # 4.5% — placeholder; confirm from competition brief

# Sensitivity of yield to expected rate change.
# y_t = Y_0 + BETA_Y × E[Δr]
# E[Δr] is in basis points (±25). With BETA_Y = 0.0001 a 25bp expected hike
# shifts yield by 2.5bp. Calibrate so the yield range matches historical data.
BETA_Y: float = 0.0001

# CPI surprise sensitivity.
# When actual CPI deviates from forecast, we linearly shift the hike/cut
# probability mass. CPI_SENSITIVITY controls how large that shift is per
# unit of surprise (e.g. per 0.1% CPI beat).
# Positive surprise (actual > forecast) → shift mass toward hike.
# Negative surprise → shift mass toward cut.
# Start conservative (0.1) and increase if the model reacts too slowly.
CPI_SENSITIVITY: float = 0.1


class YieldModel:
    """
    Derives the current yield y_t from Fed prediction market prices and CPI news.

    Pipeline:
      1. Prediction market mids for R_HIKE, R_HOLD, R_CUT give raw probabilities.
      2. Normalise to sum-to-one (market prices drift slightly off 1.0).
      3. E[Δr] = 25×q_hike − 25×q_cut  (in basis points)
      4. y_t = y_0 + β_y × E[Δr]
      5. CPI surprise adjusts the raw probability estimates between market updates.

    Consumers (StockAModel, StockCModel) read `y_t` and `delta_y` directly.
    """

    def __init__(
        self,
        y_0: float = Y_0,
        beta_y: float = BETA_Y,
        cpi_sensitivity: float = CPI_SENSITIVITY,
    ):
        self.y_0 = y_0
        self.beta_y = beta_y
        self.cpi_sensitivity = cpi_sensitivity

        # Raw (unnormalised) probability estimates for each Fed outcome.
        # Initialised to equal-weight (no prior view).
        # These are updated by two sources:
        #   - update_from_market(): reads prediction market mids each tick.
        #   - update_from_cpi():    adjusts on CPI print news events.
        self._raw_q_hike: float = 1 / 3
        self._raw_q_hold: float = 1 / 3
        self._raw_q_cut:  float = 1 / 3

        # Last CPI-based adjustment (additive delta applied on top of market mids).
        # Persists between market updates so a CPI surprise isn't lost the moment
        # the next book tick arrives. Reset to zero on the next CPI print.
        self._cpi_adjustment: float = 0.0   # positive → shift toward hike

        # Derived quantities — None until first update.
        self.q_hike: Optional[float] = None
        self.q_hold: Optional[float] = None
        self.q_cut:  Optional[float] = None
        self.E_delta_r: Optional[float] = None   # expected rate change in basis points
        self.y_t: Optional[float] = None          # current yield
        self.delta_y: Optional[float] = None      # y_t - y_0 (used by bond model)

    # ------------------------------------------------------------------
    # Update path 1: prediction market book tick
    # ------------------------------------------------------------------

    def update_from_market(self, market_state: MarketState) -> None:
        """
        Recompute y_t using the latest prediction market mids.

        Called from bot_handle_book_update whenever R_HIKE, R_HOLD, or R_CUT
        book changes. Reads weighted_mid (preferred) or mid as the price proxy.

        Prediction market prices are in [0, 100] representing cents on the
        dollar for a $1 payout, so a mid of 35 means ~35% implied probability.
        We divide by 100 to convert to a [0, 1] probability, then normalise
        so the three outcomes sum to 1.0 even if the book has drifted slightly.
        """
        mid_hike = _best_mid(market_state, "R_HIKE")
        mid_hold = _best_mid(market_state, "R_HOLD")
        mid_cut  = _best_mid(market_state, "R_CUT")

        if mid_hike is None or mid_hold is None or mid_cut is None:
            # Not all markets have quotes yet — skip this tick.
            return

        # Convert prices → raw probabilities (divide by 100 since prices are
        # in "cents per dollar"). Apply the stored CPI adjustment to hike/cut.
        raw_hike = mid_hike / 100.0 + self._cpi_adjustment
        raw_cut  = mid_cut  / 100.0 - self._cpi_adjustment
        raw_hold = mid_hold / 100.0
        # Clamp to avoid negative probabilities if the CPI shift is large.
        raw_hike = max(raw_hike, 0.0)
        raw_cut  = max(raw_cut,  0.0)
        raw_hold = max(raw_hold, 0.0)

        self._raw_q_hike = raw_hike
        self._raw_q_hold = raw_hold
        self._raw_q_cut  = raw_cut

        self._recompute()

    # ------------------------------------------------------------------
    # Update path 2: CPI structured news
    # ------------------------------------------------------------------

    def update_from_cpi(self, forecast: float, actual: float) -> None:
        """
        Adjust Fed expectations based on a CPI print surprise.

        Called from bot_handle_news when news_type == "structured" and
        structured_subtype == "cpi_print".

        surprise > 0: inflation hotter than expected → more likely to hike.
        surprise < 0: inflation cooler than expected → more likely to cut.

        The adjustment is stored as a persistent offset (_cpi_adjustment) that
        gets layered on top of the next market mid read. This means the signal
        persists even if the prediction market hasn't reacted yet — which is
        intentional, since we want to front-run the market's repricing.

        Overwrites any prior CPI adjustment (each new print supersedes the last).
        """
        surprise = actual - forecast   # e.g. +0.2 → 20bp beat

        # Linear scaling: each unit of surprise shifts hike probability by
        # cpi_sensitivity. Cap the adjustment at ±0.3 to prevent extreme
        # positions from a single data point.
        adjustment = self.cpi_sensitivity * surprise
        self._cpi_adjustment = max(-0.3, min(0.3, adjustment))

        # Immediately recompute y_t using current raw probabilities + new adjustment.
        self._recompute()

    # ------------------------------------------------------------------
    # Internal recomputation
    # ------------------------------------------------------------------

    def _recompute(self) -> None:
        """
        Normalise probabilities and derive E[Δr] and y_t.

        Normalisation: the three prediction market prices should sum to 1.0 in
        theory, but in practice the order books drift (bid-ask spread, stale
        quotes). We normalise to enforce the constraint and get clean probabilities.
        If the total is zero (all markets are empty) we bail out safely.
        """
        total = self._raw_q_hike + self._raw_q_hold + self._raw_q_cut
        if total <= 0:
            return  # degenerate; wait for valid quotes

        self.q_hike = self._raw_q_hike / total
        self.q_hold = self._raw_q_hold / total
        self.q_cut  = self._raw_q_cut  / total

        # Expected rate change in basis points.
        # Rate hike = +25bp, hold = 0bp, cut = −25bp.
        self.E_delta_r = 25.0 * self.q_hike + 0.0 * self.q_hold - 25.0 * self.q_cut

        # Current yield derived from expected rate change.
        self.y_t = self.y_0 + self.beta_y * self.E_delta_r

        # Delta_y is used directly by the bond portfolio model in StockCModel.
        self.delta_y = self.y_t - self.y_0

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Returns True once at least one full update has been computed."""
        return self.y_t is not None

    def __repr__(self) -> str:
        return (
            f"YieldModel(y_t={self.y_t:.4f}, E_delta_r={self.E_delta_r:.2f}bp, "
            f"q_hike={self.q_hike:.2%}, q_hold={self.q_hold:.2%}, q_cut={self.q_cut:.2%})"
            if self.is_ready() else "YieldModel(not yet initialised)"
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 3 — Stock A fair value
# ---------------------------------------------------------------------------

# Baseline P/E ratio at the reference yield y_0.
# Confirmed from competition parameters; update before going live.
PE_0_A: float = 15.0

# Rate sensitivity of the P/E multiple.
# Higher gamma → P/E compresses more aggressively when yields rise.
# PE_t = PE_0 × exp(−gamma × delta_y)
# With gamma=10 and a 5bp yield shock: PE drops by ~0.5%.
# With gamma=100 and a 5bp shock: PE drops by ~5%. Calibrate carefully.
GAMMA_A: float = 10.0


class StockAModel:
    """
    Fair value model for Stock A.

    Stock A is a small-cap stock priced purely on earnings:
        FV_A = EPS_A × PE_t
        PE_t = PE_0 × exp(−gamma × delta_y)

    Two update triggers:
      - on_earnings(value) : called immediately when a structured earnings
                             news event arrives for asset "A". Spec guarantees
                             two earnings prints per day.
      - on_yield_change()  : called whenever YieldModel.y_t changes
                             (i.e. after every prediction market tick or CPI
                             print). Recomputes PE_t and therefore FV_A without
                             a new EPS value.

    Because both EPS and PE_t feed into FV_A, we recompute on either change.
    This means FV_A is always current: a yield spike between earnings events
    still gets reflected immediately.
    """

    def __init__(
        self,
        yield_model: "YieldModel",
        pe_0: float = PE_0_A,
        gamma: float = GAMMA_A,
    ):
        self.yield_model = yield_model   # shared reference — reads y_t / delta_y
        self.pe_0 = pe_0
        self.gamma = gamma

        self.eps: Optional[float] = None    # most recent EPS from earnings news
        self.pe_t: Optional[float] = None   # current rate-adjusted P/E
        self.fair_value: Optional[float] = None

    # ------------------------------------------------------------------
    # Update path 1: earnings news
    # ------------------------------------------------------------------

    def on_earnings(self, eps: float) -> None:
        """
        Record a new EPS value and recompute fair value.

        Called from bot_handle_news when:
            news_type == "structured"
            structured_subtype == "earnings"
            asset == "A"

        EPS is the primary driver for Stock A. We recompute immediately so
        the bot can act before the market has fully repriced the news.
        """
        self.eps = eps
        self._recompute()

    # ------------------------------------------------------------------
    # Update path 2: yield changed (called by bot after YieldModel update)
    # ------------------------------------------------------------------

    def on_yield_change(self) -> None:
        """
        Recompute PE_t and fair_value after yield has moved.

        The YieldModel is a shared object; this method reads its current
        delta_y. Should be called after every successful YieldModel._recompute()
        — i.e. after update_from_market() or update_from_cpi() completes.

        If EPS has not yet been received this tick (eps is None) we skip
        the recompute and wait — better to have no estimate than a stale one.
        """
        self._recompute()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute(self) -> None:
        """
        Core computation:
            PE_t       = PE_0 × exp(−gamma × delta_y)
            fair_value = EPS × PE_t

        PE_t formula derivation:
            The P/E multiple is inversely related to yields — higher yields
            mean future earnings are discounted more heavily, compressing
            the multiple. The exponential form ensures PE_t is always
            positive and gives a smooth, continuous response to yield moves
            (as opposed to a linear model that could go negative).

            At delta_y = 0: PE_t = PE_0 (baseline).
            At delta_y > 0 (rates up): exp term < 1 → PE_t < PE_0 (compression).
            At delta_y < 0 (rates down): exp term > 1 → PE_t > PE_0 (expansion).
        """
        # delta_y is None until YieldModel receives its first market quote.
        # Fall back to 0.0 so that an early earnings print still produces
        # FV_A = EPS × PE_0 rather than returning None entirely.
        delta_y: float = self.yield_model.delta_y or 0.0

        pe_t = self.pe_0 * math.exp(-self.gamma * delta_y)
        self.pe_t = pe_t

        if self.eps is not None:
            self.fair_value = self.eps * pe_t

    def is_ready(self) -> bool:
        """True once both EPS and PE_t are available."""
        return self.fair_value is not None

    def mispricing(self, market_price: float) -> Optional[float]:
        """
        Returns fair_value − market_price.
        Positive → stock is underpriced (buy signal).
        Negative → stock is overpriced (sell signal).
        Returns None if fair value is not yet computed.
        """
        if self.fair_value is None:
            return None
        return self.fair_value - market_price

    def __repr__(self) -> str:
        return (
            f"StockAModel(FV={self.fair_value:.2f}, EPS={self.eps}, "
            f"PE_t={self.pe_t:.3f})"
            if self.is_ready() else "StockAModel(not yet initialised)"
        )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _best_mid(market_state: MarketState, symbol: str) -> Optional[float]:
    """
    Return the best available mid price for a symbol.

    Prefers weighted_mid (more informative) but falls back to plain mid
    if weighted_mid is unavailable (e.g. only one side of the book exists).
    """
    wm = market_state.weighted_mid(symbol)
    if wm is not None:
        return wm
    return market_state.mid(symbol)
