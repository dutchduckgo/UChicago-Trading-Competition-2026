from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Window sizes (in ticks)
# 5 ticks/second × 90 seconds/day → 450 ticks/day.
# SHORT_WINDOW = 20 ticks ≈ 4 seconds of recent price action.
# LONG_WINDOW  = 100 ticks ≈ 20 seconds — baseline "normal" volatility.
# vol_ratio = sigma_short / sigma_long > 1 signals a volatility spike,
# which we use downstream to widen quotes or reduce position size.
# ---------------------------------------------------------------------------
SHORT_WINDOW = 20   # ticks
LONG_WINDOW = 100   # ticks


@dataclass
class SymbolState:
    """
    All derived market metrics for a single symbol.

    Fields are None until enough data has arrived to compute them — callers
    must guard against None before using any value in trading logic.

    Populated by two update paths:
      - update_book()  : called on every BookSnapshot / BookUpdate event
      - update_trade() : called on every TradeMessage event
    """

    # ------------------------------------------------------------------
    # Book-derived metrics (updated on every book event)
    # ------------------------------------------------------------------

    # Simple midpoint: (best_bid + best_ask) / 2
    # Used as a quick reference price when book is balanced.
    mid: Optional[float] = None

    # Volume-weighted midpoint: (best_bid × ask_size + best_ask × bid_size)
    #                           / (bid_size + ask_size)
    # Tilts toward the side with more resting liquidity. When ask_size is
    # large, more sellers are waiting → weighted_mid is pulled toward the
    # bid, signalling potential downward pressure. More informative than
    # simple mid for short-term direction.
    weighted_mid: Optional[float] = None

    # Order book imbalance: (bid_size - ask_size) / (bid_size + ask_size)
    # Range [-1, 1]. Positive → more buying interest at the top of book.
    # Used to skew quotes: high imbalance → we're more willing to sell.
    imbalance: Optional[float] = None

    # Most recent trade price (from TradeMessage). Separate from book
    # metrics because trades can occur away from the best bid/ask.
    last_trade: Optional[float] = None

    # Best bid and ask prices from the top of book.
    # Cached here so that arb profit calculations in models.py can read
    # them directly from MarketState without accessing XChangeClient.order_books.
    # Updated every time update_book() is called.
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None

    # ------------------------------------------------------------------
    # Rolling price and return history (internal, for vol computation)
    # ------------------------------------------------------------------

    # Stores the last LONG_WINDOW+1 trade prices. We keep one extra so
    # that we can always compute a return from the previous price without
    # a separate "previous price" variable.
    # deque with maxlen auto-drops the oldest value on append — O(1) cost,
    # no manual pruning needed.
    _prices: deque = field(default_factory=lambda: deque(maxlen=LONG_WINDOW + 1))

    # Log-style simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
    # Stored separately so volatility windows slice directly into this
    # without re-deriving returns each time.
    _returns: deque = field(default_factory=lambda: deque(maxlen=LONG_WINDOW))

    # ------------------------------------------------------------------
    # Volatility metrics (updated after each new trade)
    # ------------------------------------------------------------------

    # Standard deviation of returns over the last SHORT_WINDOW ticks.
    # Reflects recent, possibly elevated, market activity.
    sigma_short: Optional[float] = None

    # Standard deviation of returns over the full LONG_WINDOW ticks.
    # Baseline "normal" volatility for this symbol.
    sigma_long: Optional[float] = None

    # sigma_short / sigma_long. > 1 → short-term vol is elevated.
    # Used as a regime indicator: spike in vol_ratio suggests a news
    # event or large order flow, warranting tighter risk controls.
    vol_ratio: Optional[float] = None

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_book(self, bids: dict, asks: dict) -> None:
        """
        Recompute mid, weighted_mid, and imbalance from the current book.

        bids / asks are dicts of {price: size} as maintained by XChangeClient.
        We only look at the best bid and best ask (top of book) — deeper
        levels are not used here to keep the hot path fast.
        """
        if not bids or not asks:
            # One side is empty (e.g. at open before any orders arrive).
            # Leave existing metrics unchanged rather than zeroing them.
            return

        best_bid = max(bids)   # highest price someone is willing to buy at
        best_ask = min(asks)   # lowest price someone is willing to sell at
        bid_size = bids[best_bid]
        ask_size = asks[best_ask]

        # Cache raw best bid/ask for arb profit calculations.
        self.best_bid = float(best_bid)
        self.best_ask = float(best_ask)
        self.mid = (best_bid + best_ask) / 2.0

        total = bid_size + ask_size
        if total > 0:
            # Weighted mid tilts toward the side with less liquidity:
            # if ask_size >> bid_size, the ask dominates the denominator
            # but bid_size scales the ask price — net effect pulls mid toward bid.
            self.weighted_mid = (best_bid * ask_size + best_ask * bid_size) / total
            self.imbalance = (bid_size - ask_size) / total
        else:
            # Degenerate case: both sides posted zero size. Fall back to mid.
            self.weighted_mid = self.mid
            self.imbalance = 0.0

    def update_trade(self, price: float) -> None:
        """
        Record a new trade price and recompute all volatility estimates.

        Called on every TradeMessage for this symbol. The exchange sends
        trade messages for all participants' fills, so this gives us the
        full tape, not just our own trades.
        """
        self.last_trade = price
        self._prices.append(price)

        # Compute return only once we have at least two prices.
        # Guard against zero previous price (shouldn't happen in practice
        # but prevents a division-by-zero if the feed has bad data).
        if len(self._prices) >= 2:
            prev = self._prices[-2]
            if prev > 0:
                r = (price - prev) / prev
                self._returns.append(r)

        self._recompute_vol()

    def _recompute_vol(self) -> None:
        """
        Recompute sigma_short, sigma_long, and vol_ratio from stored returns.

        We snapshot _returns into a plain list once so that slicing [-SHORT_WINDOW:]
        is a single O(k) operation rather than two deque traversals.
        Requires at least 2 returns for a meaningful standard deviation.
        """
        returns = list(self._returns)
        n = len(returns)
        if n < 2:
            return  # not enough history yet; leave previous estimates intact

        # Long-window vol uses all available returns (up to LONG_WINDOW).
        self.sigma_long = _std(returns)

        # Short-window vol uses only the most recent SHORT_WINDOW returns.
        short = returns[-SHORT_WINDOW:]
        if len(short) >= 2:
            self.sigma_short = _std(short)

        # vol_ratio is only meaningful when the long baseline is non-zero.
        if self.sigma_long and self.sigma_long > 0 and self.sigma_short is not None:
            self.vol_ratio = self.sigma_short / self.sigma_long


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _std(values: list) -> float:
    """
    Sample standard deviation (Bessel-corrected, divides by n-1).

    Pure Python to avoid a numpy import on the hot path. For window sizes
    of 20–100 elements this is fast enough; benchmark before switching to
    numpy if profiling shows this is a bottleneck.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# MarketState — top-level container used by the bot
# ---------------------------------------------------------------------------

class MarketState:
    """
    Owns a SymbolState for every tracked instrument.

    The bot holds a single MarketState instance and calls:
      - on_book_update(symbol, bids, asks)  from bot_handle_book_update
      - on_trade(symbol, price)             from bot_handle_trade_msg

    All other modules (models, risk) read state through get(symbol) or
    the convenience accessors (mid, weighted_mid, etc.) to avoid coupling
    directly to the internal SymbolState dataclass.
    """

    def __init__(self, symbols: list[str]):
        # Pre-populate entries for all known symbols at startup so that
        # the first book update doesn't trigger a dict insertion on the
        # hot path.
        self._state: dict[str, SymbolState] = {sym: SymbolState() for sym in symbols}

    def _ensure(self, symbol: str) -> SymbolState:
        """Lazily create a SymbolState for any symbol not seen at init."""
        if symbol not in self._state:
            self._state[symbol] = SymbolState()
        return self._state[symbol]

    # ------------------------------------------------------------------
    # Write path — called from bot event handlers
    # ------------------------------------------------------------------

    def on_book_update(self, symbol: str, bids: dict, asks: dict) -> None:
        """Update book-derived metrics for symbol."""
        self._ensure(symbol).update_book(bids, asks)

    def on_trade(self, symbol: str, price: float) -> None:
        """Record a trade and refresh volatility estimates for symbol."""
        self._ensure(symbol).update_trade(price)

    # ------------------------------------------------------------------
    # Read path — called from models and trading logic
    # ------------------------------------------------------------------

    def get(self, symbol: str) -> SymbolState:
        """Return the full SymbolState for direct field access."""
        return self._ensure(symbol)

    def mid(self, symbol: str) -> Optional[float]:
        return self._state.get(symbol, SymbolState()).mid

    def weighted_mid(self, symbol: str) -> Optional[float]:
        return self._state.get(symbol, SymbolState()).weighted_mid

    def imbalance(self, symbol: str) -> Optional[float]:
        return self._state.get(symbol, SymbolState()).imbalance

    def sigma_short(self, symbol: str) -> Optional[float]:
        return self._state.get(symbol, SymbolState()).sigma_short

    def sigma_long(self, symbol: str) -> Optional[float]:
        return self._state.get(symbol, SymbolState()).sigma_long

    def vol_ratio(self, symbol: str) -> Optional[float]:
        return self._state.get(symbol, SymbolState()).vol_ratio

    def best_bid(self, symbol: str) -> Optional[float]:
        """Best (highest) bid price at the top of the book."""
        return self._state.get(symbol, SymbolState()).best_bid

    def best_ask(self, symbol: str) -> Optional[float]:
        """Best (lowest) ask price at the top of the book."""
        return self._state.get(symbol, SymbolState()).best_ask
