"""
test_order_roundtrip.py — Place a limit order then cancel it.

Usage:
    python3 test_order_roundtrip.py <host:port> <username> <password>

The test:
  1. Connects to the exchange and waits for a book snapshot on symbol A.
  2. Places a single BUY limit order 50 ticks below the best bid (very
     unlikely to fill, so it is safe to place during live trading).
  3. Confirms the order appears in open_orders.
  4. Cancels the order.
  5. Confirms the cancel callback fires with success=True.
  6. Exits cleanly with a PASS / FAIL summary.

If no book arrives within 10 s (exchange not running), the test times out.
"""

import asyncio
import sys
from typing import Optional
from utcxchangelib import XChangeClient, Side


SYMBOL   = "A"
OFFSET   = 50          # ticks below best bid — safe, won't fill
QTY      = 1
TIMEOUT  = 15          # seconds to wait for book / cancel confirmation


class OrderRoundtripTest(XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self._got_book   = asyncio.Event()
        self._order_id: Optional[str] = None
        self._cancel_ok: Optional[bool] = None
        self._cancel_event = asyncio.Event()
        self._results: list[str] = []

    # ------------------------------------------------------------------ #
    # Event handlers                                                       #
    # ------------------------------------------------------------------ #

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol == SYMBOL and not self._got_book.is_set():
            bids = self.order_books[symbol].bids
            if bids:
                self._got_book.set()

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        self._results.append(f"  [WARN] order {order_id} filled {qty}@{price} before cancel")

    async def bot_handle_cancel_response(
        self, order_id: str, success: bool, error: Optional[str]
    ) -> None:
        if order_id == self._order_id:
            self._cancel_ok = success
            self._cancel_event.set()
            if not success:
                self._results.append(f"  [FAIL] cancel rejected: {error}")

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        if order_id == self._order_id:
            self._results.append(f"  [FAIL] order rejected: {reason}")

    async def bot_handle_trade_msg(self, symbol, price, qty): pass
    async def bot_handle_swap_response(self, swap, qty, success): pass
    async def bot_handle_news(self, news_release): pass
    async def bot_handle_market_resolved(self, market_id, winning_symbol, tick): pass
    async def bot_handle_settlement_payout(self, user, market_id, amount, tick): pass

    # ------------------------------------------------------------------ #
    # Test runner                                                          #
    # ------------------------------------------------------------------ #

    async def run_test(self):
        print(f"Connecting to {self.host} as {self.username} ...")

        # Start the exchange connection in the background.
        connect_task = asyncio.create_task(self.connect())

        # ── Step 1: wait for a live book on symbol A ──────────────────── #
        print(f"Waiting up to {TIMEOUT}s for a book snapshot on {SYMBOL} ...")
        try:
            await asyncio.wait_for(self._got_book.wait(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            print("FAIL — no book arrived within timeout. Is the exchange running?")
            connect_task.cancel()
            return

        bids = self.order_books[SYMBOL].bids
        best_bid = max(bids)
        limit_px = best_bid - OFFSET
        print(f"  Book received. best_bid={best_bid}  limit_px={limit_px}")

        # ── Step 2: place the order ───────────────────────────────────── #
        self._order_id = await self.place_order(SYMBOL, QTY, Side.BUY, limit_px)
        print(f"  Placed BUY {QTY} {SYMBOL} @ {limit_px}  (order_id={self._order_id})")

        await asyncio.sleep(0.5)   # give exchange time to acknowledge

        # ── Step 3: verify it's in open_orders ───────────────────────── #
        if self._order_id in self.open_orders:
            print(f"  PASS — order {self._order_id} is in open_orders")
            self._results.append("PASS: order acknowledged by exchange")
        else:
            print(f"  FAIL — order {self._order_id} not found in open_orders")
            self._results.append("FAIL: order missing from open_orders")

        # ── Step 4: cancel it ─────────────────────────────────────────── #
        await self.cancel_order(self._order_id)
        print(f"  Cancel request sent for {self._order_id}")

        # ── Step 5: wait for cancel confirmation ─────────────────────── #
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            print("FAIL — cancel confirmation not received within timeout.")
            self._results.append("FAIL: cancel confirmation timed out")
            connect_task.cancel()
            return

        if self._cancel_ok:
            print(f"  PASS — order {self._order_id} cancelled successfully")
            self._results.append("PASS: order cancelled successfully")
        else:
            print(f"  FAIL — cancel was rejected")

        # ── Summary ──────────────────────────────────────────────────── #
        print()
        print("=" * 50)
        for line in self._results:
            print(line)
        overall = "PASS" if all(r.startswith("PASS") for r in self._results) else "FAIL"
        print(f"Overall: {overall}")
        print("=" * 50)

        connect_task.cancel()

    async def start(self):
        await self.run_test()


async def main():
    if len(sys.argv) != 4:
        print("Usage: python3 test_order_roundtrip.py <host:port> <username> <password>")
        sys.exit(1)

    host, username, password = sys.argv[1], sys.argv[2], sys.argv[3]
    client = OrderRoundtripTest(host, username, password)
    try:
        await client.start()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
