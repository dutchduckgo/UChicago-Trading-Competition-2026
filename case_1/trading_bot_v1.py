from utcxchangelib import XChangeClient, Side
from typing import Optional
import asyncio


class MyXchangeClient(XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        print("order fill", self.positions)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("order rejected because of", reason)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        pass

    async def bot_handle_news(self, news_release: dict):
        tick = news_release["tick"]
        news_type = news_release["kind"]
        symbol = news_release["symbol"]  # may be None
        news_data = news_release["new_data"]

        if news_type == "structured":
            subtype = news_data["structured_subtype"]
            if subtype == "earnings":
                asset = news_data["asset"]
                value = news_data["value"]
            elif subtype == "cpi_print":
                forecast = news_data["forecast"]
                actual = news_data["actual"]
        else:
            content = news_data["content"]
            message_type = news_data["type"]

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        print(f"Market {market_id} resolved: winner is {winning_symbol}")

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        print(f"Settlement payout: {amount} from {market_id}")

    async def trade(self):
        await asyncio.sleep(5)
        # Your trading logic here

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()

async def main():
    SERVER = "SERVER_HOST:3333"
    my_client = MyXchangeClient(SERVER, "USERNAME", "PASSWORD")
    await my_client.start()


if __name__ == "__main__":
    asyncio.run(main())