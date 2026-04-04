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
        self.E_delta_r = 25.0 * self.q_hike - 25.0 * self.q_cut # removed + 0.0 * self.q_hold

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
# Phase 4 — Stock C fair value
# ---------------------------------------------------------------------------

# Stock C P/E parameters — same exponential form as Stock A.
# Stock C is a large insurance company so its P/E may compress differently
# than a small-cap; keep GAMMA_C as a separate knob even if initially equal.
PE_0_C: float = 12.0
GAMMA_C: float = 10.0

# Bond portfolio parameters.
# These describe the insurance company's fixed-income holdings.
# B_0    : total market value of the bond portfolio at the reference yield y_0.
# D      : modified duration (years). Controls linear rate sensitivity.
#          A duration of 7 means a 100bp yield rise ≈ 7% drop in bond value.
# CONV   : convexity. Corrects for the curvature in the price-yield relationship.
#          Bonds gain more in price when yields fall than they lose when yields
#          rise by the same amount — convexity captures this asymmetry.
# N      : number of shares outstanding. Scales ΔB down to a per-share contribution.
# LAMBDA : weighting constant that maps the per-share bond PnL into a stock price
#          impact. Reflects how much of the bond portfolio change flows through
#          to the stock price (accounting for leverage, hedges, etc.).
B_0:    float = 1_000_000.0   # placeholder — confirm from competition parameters
D:      float = 7.0            # years of duration
CONV:   float = 60.0           # convexity (dimensionless)
N:      float = 10_000.0       # shares outstanding
LAMBDA: float = 1.0            # pass-through weight; calibrate against C price history


class StockCModel:
    """
    Fair value model for Stock C (large insurance company).

    Stock C has two price components:

        FV_C = operations_value + bond_portfolio_impact + noise

    where:
        operations_value     = EPS_C × PE_t
        bond_portfolio_impact = lambda × (ΔB / N)
        ΔB ≈ B_0 × (−D × Δy  +  0.5 × Conv × Δy²)

    The noise term is unmodellable; we ignore it and trade on the signal
    from the two structured components.

    Update triggers (same pattern as StockAModel):
      - on_earnings(eps) : new EPS from structured earnings news for asset "C"
      - on_yield_change(): yield has moved — recomputes both PE_t and ΔB
    """

    def __init__(
        self,
        yield_model: "YieldModel",
        pe_0: float = PE_0_C,
        gamma: float = GAMMA_C,
        b_0: float = B_0,
        duration: float = D,
        convexity: float = CONV,
        n_shares: float = N,
        lam: float = LAMBDA,
    ):
        self.yield_model = yield_model
        self.pe_0 = pe_0
        self.gamma = gamma
        self.b_0 = b_0
        self.duration = duration
        self.convexity = convexity
        self.n_shares = n_shares
        self.lam = lam

        self.eps: Optional[float] = None

        # Decomposed components — stored separately so downstream code can
        # inspect how much of the fair value comes from operations vs bonds.
        self.pe_t: Optional[float] = None
        self.operations_value: Optional[float] = None   # EPS × PE_t
        self.delta_b: Optional[float] = None            # total bond portfolio ΔV
        self.bond_impact: Optional[float] = None        # lambda × (ΔB / N) per share
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
            asset == "C"
        """
        self.eps = eps
        self._recompute()

    # ------------------------------------------------------------------
    # Update path 2: yield changed
    # ------------------------------------------------------------------

    def on_yield_change(self) -> None:
        """
        Recompute PE_t, ΔB, and fair_value after yield has moved.

        Both the operations component (via PE_t) and the bond component
        (via ΔB) depend on delta_y, so a single yield move updates both
        simultaneously. This is the key difference from Stock A — for C,
        yield changes have a direct, immediate second path to price through
        the bond portfolio, making it more rate-sensitive overall.
        """
        self._recompute()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute(self) -> None:
        """
        Core computation:

        Step 1 — Rate-adjusted P/E (identical to StockAModel):
            PE_t = PE_0 × exp(−gamma × delta_y)

        Step 2 — Bond portfolio value change (Taylor expansion):
            ΔB ≈ B_0 × (−D × Δy  +  0.5 × Conv × Δy²)

            The first-order term (−D × Δy) is the dominant effect:
              yields up → bond prices down.
            The second-order term (0.5 × Conv × Δy²) is always positive:
              convexity means bonds lose less than duration predicts on
              yield rises, and gain more on yield falls.

        Step 3 — Per-share bond impact:
            bond_impact = lambda × (ΔB / N)

        Step 4 — Combined fair value:
            FV_C = (EPS × PE_t) + bond_impact
                 = operations_value + bond_impact
        """
        delta_y: float = self.yield_model.delta_y or 0.0

        # Step 1: P/E with rate adjustment
        pe_t = self.pe_0 * math.exp(-self.gamma * delta_y)
        self.pe_t = pe_t

        # Step 2: Bond portfolio change via duration-convexity approximation.
        # Sign convention: delta_y > 0 (yields rise) → ΔB negative (bonds lose value).
        self.delta_b = self.b_0 * (
            -self.duration * delta_y
            + 0.5 * self.convexity * delta_y ** 2
        )

        # Step 3: Per-share bond impact scaled by lambda.
        self.bond_impact = self.lam * (self.delta_b / self.n_shares)

        # Step 4: Combine only once we have EPS. Bond impact is always computed
        # (it depends only on yield, not EPS) so we can still track rate
        # sensitivity even before the first earnings print.
        if self.eps is not None:
            self.operations_value = self.eps * pe_t
            self.fair_value = self.operations_value + self.bond_impact

    def is_ready(self) -> bool:
        """True once EPS has been received and fair_value is computed."""
        return self.fair_value is not None

    def mispricing(self, market_price: float) -> Optional[float]:
        """
        Returns fair_value − market_price.
        Positive → underpriced (buy signal).
        Negative → overpriced (sell signal).
        Returns None if fair value is not yet computed.
        """
        if self.fair_value is None:
            return None
        return self.fair_value - market_price

    def __repr__(self) -> str:
        if not self.is_ready():
            return "StockCModel(not yet initialised)"
        return (
            f"StockCModel(FV={self.fair_value:.2f}, "
            f"ops={self.operations_value:.2f}, "
            f"bond_impact={self.bond_impact:.2f}, "
            f"PE_t={self.pe_t:.3f}, "
            f"ΔB={self.delta_b:.0f})"
        )


# ---------------------------------------------------------------------------
# Phase 5 — ETF arbitrage model
# ---------------------------------------------------------------------------

# Flat swap cost (both directions) as defined in DEFAULT_SWAP_MAP.
# toETF:   A + B + C → ETF, cost 5
# fromETF: ETF → A + B + C, cost 5
ETF_SWAP_COST: int = 5


class ETFModel:
    """
    Fair value and arbitrage model for ETF AKAV.

    AKAV = 1 share A + 1 share B + 1 share C.
    Fair value: FV_ETF = FV_A + FV_B + FV_C

    Stock B has no fundamental model, so we use its market mid as FV_B.
    This means the ETF FV estimate inherits whatever noise exists in the B mid,
    but there is no better alternative without an options-derived estimate.

    Two types of arbitrage, both requiring a swap (cost 5 flat):

      SELL ETF arb (ETF overpriced):
        Buy A at ask, buy B at ask, buy C at ask
        → place_swap_order("toETF")          [cost 5]
        → sell ETF at bid
        Profit = ETF_bid − (ask_A + ask_B + ask_C) − 5

      BUY ETF arb (ETF underpriced):
        Buy ETF at ask
        → place_swap_order("fromETF")        [cost 5]
        → sell A at bid, sell B at bid, sell C at bid
        Profit = (bid_A + bid_B + bid_C) − ETF_ask − 5

    The model exposes both the FV-based mispricing (a noisy, model-dependent
    signal) and the executable arb profit (derived entirely from live book
    prices — model-independent). The bot should gate on executable profit
    being positive before sending swap orders.
    """

    def __init__(
        self,
        stock_a: StockAModel,
        stock_c: StockCModel,
        market_state: MarketState,
        swap_cost: int = ETF_SWAP_COST,
    ):
        self.stock_a = stock_a
        self.stock_c = stock_c
        self.market_state = market_state
        self.swap_cost = swap_cost

        # Cached fair values — updated on demand via update()
        self.fv_a: Optional[float] = None
        self.fv_b: Optional[float] = None   # market mid — no fundamental model
        self.fv_c: Optional[float] = None
        self.fair_value: Optional[float] = None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> None:
        """
        Recompute FV_ETF from current model estimates and market mids.

        Call this after any event that could change component fair values:
          - After on_earnings / on_yield_change for A or C
          - After a book update for B or ETF
          - After any YieldModel update

        FV_B uses weighted_mid (falls back to mid) as the best available
        price for B since no earnings or structural model exists for it.
        """
        self.fv_a = self.stock_a.fair_value
        self.fv_b = _best_mid(self.market_state, "B")
        self.fv_c = self.stock_c.fair_value

        if self.fv_a is not None and self.fv_b is not None and self.fv_c is not None:
            self.fair_value = self.fv_a + self.fv_b + self.fv_c

    def is_ready(self) -> bool:
        return self.fair_value is not None

    # ------------------------------------------------------------------
    # Signal 1: model-based mispricing
    # ------------------------------------------------------------------

    def fv_mispricing(self) -> Optional[float]:
        """
        FV_ETF − market_mid(ETF).

        Positive → ETF is underpriced relative to component fair values (buy ETF).
        Negative → ETF is overpriced (sell ETF / buy components).

        This signal is noisy because FV_A and FV_C depend on model parameters
        (EPS, PE_0, bond params) that may be miscalibrated. Use it as a
        directional indicator rather than a precise threshold.
        """
        if self.fair_value is None:
            return None
        etf_mid = _best_mid(self.market_state, "ETF")
        if etf_mid is None:
            return None
        return self.fair_value - etf_mid

    # ------------------------------------------------------------------
    # Signal 2: executable arb profit (model-independent)
    # ------------------------------------------------------------------

    def sell_etf_arb_profit(self) -> Optional[float]:
        """
        Profit from: buy A+B+C at ask → toETF swap (cost 5) → sell ETF at bid.

        ETF is overpriced when this is positive.
        Returns None if any required book price is missing.

        This is model-independent: it uses only live bid/ask prices and the
        known swap cost. Execute when > 0, size conservatively on first trades
        until you have confidence in fill rates on all legs.
        """
        etf_bid = _best_bid(self.market_state, "ETF")
        ask_a   = _best_ask(self.market_state, "A")
        ask_b   = _best_ask(self.market_state, "B")
        ask_c   = _best_ask(self.market_state, "C")

        if any(x is None for x in (etf_bid, ask_a, ask_b, ask_c)):
            return None

        return etf_bid - (ask_a + ask_b + ask_c) - self.swap_cost   # type: ignore[operator]

    def buy_etf_arb_profit(self) -> Optional[float]:
        """
        Profit from: buy ETF at ask → fromETF swap (cost 5) → sell A+B+C at bid.

        ETF is underpriced when this is positive.
        Returns None if any required book price is missing.
        """
        etf_ask = _best_ask(self.market_state, "ETF")
        bid_a   = _best_bid(self.market_state, "A")
        bid_b   = _best_bid(self.market_state, "B")
        bid_c   = _best_bid(self.market_state, "C")

        if any(x is None for x in (etf_ask, bid_a, bid_b, bid_c)):
            return None

        return (bid_a + bid_b + bid_c) - etf_ask - self.swap_cost   # type: ignore[operator]

    def __repr__(self) -> str:
        if not self.is_ready():
            return "ETFModel(not yet initialised)"
        sell_p = self.sell_etf_arb_profit()
        buy_p  = self.buy_etf_arb_profit()
        return (
            f"ETFModel(FV={self.fair_value:.2f}, "
            f"sell_arb={sell_p:.2f if sell_p is not None else 'N/A'}, "
            f"buy_arb={buy_p:.2f if buy_p is not None else 'N/A'})"
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _best_bid(market_state: MarketState, symbol: str) -> Optional[float]:
    """
    Best (highest) bid price at the top of the book for symbol.
    Reads from MarketState.best_bid, which is cached by SymbolState.update_book()
    on every book snapshot and incremental update.
    """
    return market_state.best_bid(symbol)


def _best_ask(market_state: MarketState, symbol: str) -> Optional[float]:
    """
    Best (lowest) ask price at the top of the book for symbol.
    Reads from MarketState.best_ask, which is cached by SymbolState.update_book().
    """
    return market_state.best_ask(symbol)


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
