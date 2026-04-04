"""
trading_bot_v1.py — Main bot: event wiring, quote logic, risk gating.

Architecture
------------
    trading_bot_v1.py  (this file)  — event handlers, quoting, risk
    models.py                       — fair value models (A, C, yield)
    state.py                        — per-symbol book metrics & volatility

Exchange symbols
----------------
    Equities  : A, B, C
    ETF       : ETF  (1 share each of A, B, C; swap cost = 5 flat)
    Options   : B_C_950, B_P_950, B_C_1000, B_P_1000, B_C_1050, B_P_1050
    Pred mkts : R_HIKE, R_HOLD, R_CUT

Swap names (as registered in DEFAULT_SWAP_MAP)
-----------------------------------------------
    toETF   : surrender A(1) + B(1) + C(1), receive ETF(1), pay cost 5
    fromETF : surrender ETF(1), receive A(1) + B(1) + C(1), pay cost 5
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from utcxchangelib import XChangeClient, Side

from state import MarketState
from models import YieldModel, StockAModel, StockCModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol groups
# ---------------------------------------------------------------------------
ALL_SYMBOLS = [
    "A", "B", "C", "ETF",
    "B_C_950", "B_P_950", "B_C_1000", "B_P_1000", "B_C_1050", "B_P_1050",
    "R_CUT", "R_HOLD", "R_HIKE",
]
PRED_MARKET_SYMS = frozenset({"R_HIKE", "R_HOLD", "R_CUT"})
EQUITY_SYMS      = frozenset({"A", "B", "C"})
QUOTED_SYMS      = frozenset({"A", "C"})          # symbols we actively market-make
ETF_SYM          = "ETF"
SWAP_TO_ETF      = "toETF"
SWAP_FROM_ETF    = "fromETF"
ETF_SWAP_COST    = 5                               # flat fee per swap

# ---------------------------------------------------------------------------
# Market-making parameters — tune against historical data before going live
# ---------------------------------------------------------------------------

# Minimum half-spread posted around fair value (in integer price ticks).
# Below this we are not compensated for adverse selection.
BASE_HALF_SPREAD: int = 3

# How much extra spread to add per unit of long-window vol.
# half_spread += round(sigma_long * VOL_SPREAD_FACTOR)
# With sigma_long ≈ 0.001 and factor = 200 → +0 ticks (negligible).
# Increase once you know the typical vol magnitude for each symbol.
VOL_SPREAD_FACTOR: float = 200.0

# Hard position limit per symbol. Orders that would breach this are skipped.
MAX_POSITION: int = 150

# Default quote size (shares per side).
QUOTE_SIZE: int = 5

# Inventory skew: for every share of net position, shift our quoted mid by
# this many ticks against our inventory (so we lean toward getting flat).
# E.g. long 10 shares of A → shift our bid/ask centre down by 10 * 0.3 = 3 ticks.
INVENTORY_SKEW_PER_SHARE: float = 0.3

# Cancel and re-quote at most once every this many seconds.
# Prevents the bot from flooding the exchange with cancel+replace on every tick.
REQUOTE_MIN_INTERVAL: float = 0.5

# How far (in ticks) fair value must move before we force an immediate requote,
# regardless of the time-based throttle.
FV_REQUOTE_THRESHOLD: float = 1.0

# ETF arbitrage: minimum profit above swap cost to pull the trigger (ticks).
ETF_ARB_MIN_EDGE: float = 3.0

# Size for each leg of an ETF arb (number of swaps).
ETF_ARB_SIZE: int = 1


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class MyXchangeClient(XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)

        # --- Market state (per-symbol book metrics + volatility) -----------
        self.market_state = MarketState(ALL_SYMBOLS)

        # --- Fair value models --------------------------------------------
        self.yield_model  = YieldModel()
        self.stock_a      = StockAModel(self.yield_model)
        self.stock_c      = StockCModel(self.yield_model)

        # --- Active quotes: {symbol: {"bid": order_id, "ask": order_id}} --
        # None means no live quote on that side.
        self._active_quotes: dict[str, dict[str, Optional[str]]] = {
            sym: {"bid": None, "ask": None} for sym in QUOTED_SYMS
        }

        # --- Requote throttle: {symbol: last requote time (event loop time)} -
        self._last_requote_time: dict[str, float] = {sym: 0.0 for sym in QUOTED_SYMS}

        # --- Last fair value used for quoting (to detect significant moves) --
        self._last_quoted_fv: dict[str, Optional[float]] = {sym: None for sym in QUOTED_SYMS}

    # -----------------------------------------------------------------------
    # XChangeClient event overrides
    # -----------------------------------------------------------------------

    async def bot_handle_book_update(self, symbol: str) -> None:
        """
        Called on every incremental book update or full snapshot.

        1. Push new bids/asks into MarketState.
        2. If the symbol is a prediction market, recompute yields and propagate
           to rate-sensitive models, then requote Stock C if FV moved enough.
        """
        bids = self.order_books[symbol].bids
        asks = self.order_books[symbol].asks
        self.market_state.on_book_update(symbol, bids, asks)

        if symbol in PRED_MARKET_SYMS:
            self.yield_model.update_from_market(self.market_state)
            # Propagate yield change to both stock models.
            self.stock_a.on_yield_change()
            self.stock_c.on_yield_change()
            # Rate move affects C more strongly — requote immediately if FV moved.
            await self._maybe_requote("C", force_if_fv_moved=True)
            await self._maybe_requote("A", force_if_fv_moved=True)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        """Record every trade (ours and others') for volatility estimation."""
        self.market_state.on_trade(symbol, float(price))

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        """
        A fill was received. Remove from active_quotes tracking if it was
        one of our quotes, so we know that side is no longer live.
        """
        for sym, quotes in self._active_quotes.items():
            if quotes["bid"] == order_id:
                quotes["bid"] = None
                logger.info("Bid quote filled for %s: %d @ %d", sym, qty, price)
            elif quotes["ask"] == order_id:
                quotes["ask"] = None
                logger.info("Ask quote filled for %s: %d @ %d", sym, qty, price)

    async def bot_handle_cancel_response(
        self, order_id: str, success: bool, error: Optional[str]
    ) -> None:
        if not success:
            logger.warning("Cancel failed for order %s: %s", order_id, error)
        # Remove from active_quotes regardless; if cancel failed the order may
        # have already filled (handled in bot_handle_order_fill).
        for quotes in self._active_quotes.values():
            if quotes["bid"] == order_id:
                quotes["bid"] = None
            if quotes["ask"] == order_id:
                quotes["ask"] = None

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        logger.warning("Order %s rejected: %s", order_id, reason)
        for quotes in self._active_quotes.values():
            if quotes["bid"] == order_id:
                quotes["bid"] = None
            if quotes["ask"] == order_id:
                quotes["ask"] = None

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        if success:
            logger.info("Swap %s x%d succeeded", swap, qty)
        else:
            logger.warning("Swap %s x%d failed", swap, qty)

    async def bot_handle_news(self, news_release: dict):
        """
        Route structured and unstructured news to the appropriate model.

        Structured dispatch:
          earnings / asset == "A" → StockAModel.on_earnings()
          earnings / asset == "C" → StockCModel.on_earnings()
          cpi_print              → YieldModel.update_from_cpi() → propagate yields

        Unstructured:
          Logged for now; sentiment parsing goes here in a future phase.
        """
        tick      = news_release["tick"]
        news_type = news_release["kind"]
        news_data = news_release["new_data"]

        if news_type == "structured":
            subtype = news_data["structured_subtype"]

            if subtype == "earnings":
                asset = news_data["asset"]
                value = float(news_data["value"])
                logger.info("Tick %d | Earnings: asset=%s value=%.4f", tick, asset, value)

                if asset == "A":
                    self.stock_a.on_earnings(value)
                    await self._requote("A")
                elif asset == "C":
                    self.stock_c.on_earnings(value)
                    await self._requote("C")

            elif subtype == "cpi_print":
                forecast = float(news_data["forecast"])
                actual   = float(news_data["actual"])
                logger.info(
                    "Tick %d | CPI print: forecast=%.2f actual=%.2f surprise=%.2f",
                    tick, forecast, actual, actual - forecast,
                )
                self.yield_model.update_from_cpi(forecast, actual)
                self.stock_a.on_yield_change()
                self.stock_c.on_yield_change()
                # CPI is a high-signal event — requote both immediately.
                await self._requote("A")
                await self._requote("C")

        else:
            content      = news_data.get("content", "")
            message_type = news_data.get("type", "")
            logger.info("Tick %d | Unstructured news [%s]: %s", tick, message_type, content)
            # TODO: keyword/sentiment parsing to adjust FV estimates.

    async def bot_handle_market_resolved(
        self, market_id: str, winning_symbol: str, tick: int
    ):
        logger.info("Market %s resolved → %s at tick %d", market_id, winning_symbol, tick)

    async def bot_handle_settlement_payout(
        self, user: str, market_id: str, amount: int, tick: int
    ):
        logger.info("Settlement payout from %s: %d at tick %d", market_id, amount, tick)

    # -----------------------------------------------------------------------
    # Background trade loop
    # -----------------------------------------------------------------------

    async def trade(self):
        """
        Periodic background task. Runs every REQUOTE_MIN_INTERVAL seconds.

        Responsibilities:
          1. Refresh quotes on A and C (time-based cadence).
          2. Check for ETF arbitrage opportunities.
          3. Future: cancel stale orders, manage options.
        """
        await asyncio.sleep(5)  # wait for initial book snapshots

        while True:
            await asyncio.sleep(REQUOTE_MIN_INTERVAL)

            try:
                # Refresh quotes for all actively market-made symbols.
                for sym in QUOTED_SYMS:
                    await self._maybe_requote(sym, force_if_fv_moved=False)

                # ETF arbitrage check.
                await self._check_etf_arb()

            except Exception as exc:
                logger.exception("Unhandled exception in trade loop: %s", exc)

    # -----------------------------------------------------------------------
    # Quoting helpers
    # -----------------------------------------------------------------------

    async def _maybe_requote(self, symbol: str, force_if_fv_moved: bool = False) -> None:
        """
        Requote `symbol` if:
          (a) REQUOTE_MIN_INTERVAL seconds have elapsed since the last requote, OR
          (b) `force_if_fv_moved` is True AND fair value has moved by more than
              FV_REQUOTE_THRESHOLD ticks since the last requote.
        """
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_requote_time.get(symbol, 0.0)
        time_ok = elapsed >= REQUOTE_MIN_INTERVAL

        fv_moved = False
        if force_if_fv_moved:
            new_fv = self._fair_value(symbol)
            last_fv = self._last_quoted_fv.get(symbol)
            if new_fv is not None and (
                last_fv is None or abs(new_fv - last_fv) >= FV_REQUOTE_THRESHOLD
            ):
                fv_moved = True

        if time_ok or fv_moved:
            await self._requote(symbol)

    async def _requote(self, symbol: str) -> None:
        """
        Cancel any live quotes for `symbol` and post fresh bid/ask around FV.

        Steps:
          1. Compute fair value — skip if not ready.
          2. Compute half-spread (base + vol-adjusted).
          3. Compute inventory skew.
          4. Cancel existing quotes.
          5. Place new bid and ask (subject to position limits).
        """
        fv = self._fair_value(symbol)
        if fv is None:
            return

        half_spread = self._compute_half_spread(symbol)

        # Inventory skew: shift quoted centre against current net position so
        # we lean toward getting flat.
        net_pos = self.positions.get(symbol, 0)
        skew = net_pos * INVENTORY_SKEW_PER_SHARE   # positive if long → move quotes down

        bid_px = round(fv - half_spread - skew)
        ask_px = round(fv + half_spread - skew)
        # Enforce minimum tick separation.
        if ask_px <= bid_px:
            ask_px = bid_px + 1

        await self._cancel_symbol_quotes(symbol)

        # Respect hard position limits: skip a side if the fill would breach them.
        bid_size = QUOTE_SIZE
        ask_size = QUOTE_SIZE
        if net_pos + bid_size > MAX_POSITION:
            bid_size = max(0, MAX_POSITION - net_pos)
        if net_pos - ask_size < -MAX_POSITION:
            ask_size = max(0, net_pos + MAX_POSITION)

        quotes = self._active_quotes[symbol]
        if bid_size > 0:
            bid_id = await self.place_order(symbol, bid_size, Side.BUY, bid_px)
            quotes["bid"] = bid_id

        if ask_size > 0:
            ask_id = await self.place_order(symbol, ask_size, Side.SELL, ask_px)
            quotes["ask"] = ask_id

        self._last_requote_time[symbol] = asyncio.get_event_loop().time()
        self._last_quoted_fv[symbol] = fv

        logger.debug(
            "Requote %s | FV=%.2f spread=±%d skew=%.1f → bid=%d ask=%d pos=%d",
            symbol, fv, half_spread, skew, bid_px, ask_px, net_pos,
        )

    async def _cancel_symbol_quotes(self, symbol: str) -> None:
        """Cancel both the live bid and ask for `symbol` if they exist."""
        quotes = self._active_quotes.get(symbol, {})
        for side_key in ("bid", "ask"):
            oid = quotes.get(side_key)
            if oid is not None and oid in self.open_orders:
                await self.cancel_order(oid)

    # -----------------------------------------------------------------------
    # ETF arbitrage helper
    # -----------------------------------------------------------------------

    async def _check_etf_arb(self) -> None:
        """
        Look for ETF mispricing vs the sum of component fair values.

        ETF fair value  = FV_A + FV_B + FV_C
        (FV_B is approximated by the ETF mid minus FV_A minus FV_C when no
        model exists for B; until then we use the B market mid as a proxy.)

        If ETF_mid > FV_ETF + ETF_SWAP_COST + ETF_ARB_MIN_EDGE:
            Arb: acquire components, swap toETF, sell ETF.
            In practice: place_swap_order(fromETF) to get components, then sell.
            *** Simplified below — full leg management is a TODO. ***

        If ETF_mid < FV_ETF - ETF_SWAP_COST - ETF_ARB_MIN_EDGE:
            Arb: buy ETF, swap fromETF, sell components.
        """
        fv_a = self._fair_value("A")
        fv_c = self._fair_value("C")
        etf_mid = self.market_state.mid(ETF_SYM)
        b_mid   = self.market_state.mid("B")

        # Need all four prices to evaluate arb.
        if any(x is None for x in (fv_a, fv_c, etf_mid, b_mid)):
            return

        fv_etf = fv_a + b_mid + fv_c   # use B market mid as proxy for FV_B

        buy_etf_threshold  = fv_etf - ETF_SWAP_COST - ETF_ARB_MIN_EDGE
        sell_etf_threshold = fv_etf + ETF_SWAP_COST + ETF_ARB_MIN_EDGE

        if etf_mid < buy_etf_threshold:
            # ETF is cheap: buy ETF at market, decompose via fromETF swap.
            logger.info(
                "ETF arb (buy ETF): ETF mid=%.1f FV_ETF=%.1f edge=%.1f",
                etf_mid, fv_etf, buy_etf_threshold - etf_mid,
            )
            etf_pos = self.positions.get(ETF_SYM, 0)
            if etf_pos < MAX_POSITION:
                await self.place_order(ETF_SYM, ETF_ARB_SIZE, Side.BUY)
                await self.place_swap_order(SWAP_FROM_ETF, ETF_ARB_SIZE)

        elif etf_mid > sell_etf_threshold:
            # ETF is expensive: swap components toETF, sell ETF.
            a_pos = self.positions.get("A", 0)
            b_pos = self.positions.get("B", 0)
            c_pos = self.positions.get("C", 0)
            min_component = min(a_pos, b_pos, c_pos)
            if min_component >= ETF_ARB_SIZE:
                logger.info(
                    "ETF arb (sell ETF): ETF mid=%.1f FV_ETF=%.1f edge=%.1f",
                    etf_mid, fv_etf, etf_mid - sell_etf_threshold,
                )
                await self.place_swap_order(SWAP_TO_ETF, ETF_ARB_SIZE)
                await self.place_order(ETF_SYM, ETF_ARB_SIZE, Side.SELL)

    # -----------------------------------------------------------------------
    # Fair value helpers
    # -----------------------------------------------------------------------

    def _fair_value(self, symbol: str) -> Optional[float]:
        """Return the current model fair value for `symbol`, or None if not ready."""
        if symbol == "A":
            return self.stock_a.fair_value
        if symbol == "C":
            return self.stock_c.fair_value
        # For all other symbols, fall back to the market mid.
        return self.market_state.mid(symbol)

    def _compute_half_spread(self, symbol: str) -> int:
        """
        Return the half-spread to post around fair value for `symbol`.

        half_spread = BASE_HALF_SPREAD
                    + round(sigma_long * VOL_SPREAD_FACTOR)

        Falls back to BASE_HALF_SPREAD when vol is not yet estimated.
        """
        sigma = self.market_state.sigma_long(symbol) or 0.0
        fv    = self._fair_value(symbol) or 1.0
        # sigma is a return (fraction), convert to price ticks:
        vol_component = round(sigma * fv * VOL_SPREAD_FACTOR)
        return max(BASE_HALF_SPREAD, BASE_HALF_SPREAD + vol_component)

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


async def main():
    SERVER = "34.197.188.76:3333"
    my_client = MyXchangeClient(SERVER, "texas1", "yarrow-torch-gust")
    await my_client.start()


if __name__ == "__main__":
    asyncio.run(main())
