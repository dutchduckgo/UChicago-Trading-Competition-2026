# Case 1 — Implementation Context

This document summarises the architecture, current state, and remaining work for the Case 1 market-making bot. It is intended as a quick-start reference for anyone picking up the codebase mid-development.

---

## File Overview

```
case_1/
├── trading_bot_v1.py       — Main bot: exchange connection, event wiring, quoting, arb
├── models.py               — Fair value models (YieldModel, StockAModel, StockCModel)
├── state.py                — Per-symbol market state (book metrics, volatility)
├── test_order_roundtrip.py — Integration test: place + cancel a limit order
├── design.md               — Original design notes and phase breakdown
└── project_description.txt — Competition case specification
```

---

## Architecture

The bot is split into three layers:

```
trading_bot_v1.py   — XChangeClient subclass
    ├── MarketState (state.py)
    │       └── SymbolState (per symbol)
    └── Models (models.py)
            ├── YieldModel
            ├── StockAModel  (depends on YieldModel)
            └── StockCModel  (depends on YieldModel)
```

- **`state.py`** owns all raw market data — bids, asks, trades, rolling volatility. No model logic lives here.
- **`models.py`** owns fair value computation. Models read from `YieldModel` (shared reference) and update when earnings or yield changes arrive.
- **`trading_bot_v1.py`** connects the exchange event stream to the models and executes orders.

---

## Exchange Details

| Field    | Value                    |
|----------|--------------------------|
| Host     | `34.197.188.76:3333`     |
| Username | `texas1`                 |
| Password | `yarrow-torch-gust`      |

Credentials are already set in `main()` of `trading_bot_v1.py`.

**Verified:** Order place + cancel roundtrip confirmed PASS against live exchange on 2026-04-04.

### Symbols

| Symbol | Description |
|---|---|
| `A` | Small-cap stock — priced on EPS × PE |
| `B` | Semiconductor — no fundamental model; trade via options |
| `C` | Insurance company — EPS × PE + bond portfolio |
| `ETF` | 1 share each of A, B, C |
| `B_C_950/1000/1050` | Call options on B at 3 strikes |
| `B_P_950/1000/1050` | Put options on B at 3 strikes |
| `R_HIKE / R_HOLD / R_CUT` | Fed rate decision prediction markets |

### Swaps (ETF creation/redemption)

| Swap name | Direction | Cost |
|---|---|---|
| `toETF` | A(1) + B(1) + C(1) → ETF(1) | 5 flat |
| `fromETF` | ETF(1) → A(1) + B(1) + C(1) | 5 flat |

---

## What's Implemented

### `state.py` — Market State

`SymbolState` (one per symbol) tracks:

| Field | Formula |
|---|---|
| `mid` | `(best_bid + best_ask) / 2` |
| `weighted_mid` | `(best_bid × ask_size + best_ask × bid_size) / total_size` |
| `imbalance` | `(bid_size − ask_size) / (bid_size + ask_size)` |
| `sigma_short` | Sample std of last 20 returns |
| `sigma_long` | Sample std of last 100 returns |
| `vol_ratio` | `sigma_short / sigma_long` — regime indicator |

`MarketState` is the top-level container. The bot calls:
- `on_book_update(symbol, bids, asks)` — from `bot_handle_book_update`
- `on_trade(symbol, price)` — from `bot_handle_trade_msg`

---

### `models.py` — Fair Value Models

#### `YieldModel`
Derives current yield `y_t` from two sources:

1. **Prediction market mids** (`R_HIKE/HOLD/CUT`):
   ```
   E[Δr] = 25×q_hike + 0×q_hold − 25×q_cut   (basis points)
   y_t   = y_0 + β_y × E[Δr]
   ```

2. **CPI prints** (structured news):
   ```
   surprise    = actual − forecast
   adjustment  = clamp(CPI_SENSITIVITY × surprise, −0.3, +0.3)
   ```
   The adjustment persists as an offset on top of prediction market mids — intentionally front-runs the market's repricing.

#### `StockAModel`
```
PE_t  = PE_0 × exp(−γ × Δy)
FV_A  = EPS_A × PE_t
```
Updates on: earnings news for A, or any yield change.

#### `StockCModel`
```
PE_t          = PE_0 × exp(−γ × Δy)
ΔB            = B_0 × (−D×Δy + 0.5×Conv×Δy²)
bond_impact   = λ × (ΔB / N)
FV_C          = EPS_C × PE_t + bond_impact
```
Updates on: earnings news for C, or any yield change.
C is more rate-sensitive than A because yield moves hit both the P/E multiple and the bond portfolio simultaneously.

---

### `trading_bot_v1.py` — Event Wiring & Quoting

#### Event handler connections

| Event | Action |
|---|---|
| `bot_handle_book_update` | Update `MarketState`; if pred market symbol → recompute yield → propagate to models → requote if FV moved ≥ 1 tick |
| `bot_handle_trade_msg` | Feed trade into `MarketState` for vol tracking |
| `bot_handle_news` (earnings A) | `StockAModel.on_earnings()` → immediate requote A |
| `bot_handle_news` (earnings C) | `StockCModel.on_earnings()` → immediate requote C |
| `bot_handle_news` (CPI print) | `YieldModel.update_from_cpi()` → propagate → immediate requote A + C |
| `bot_handle_order_fill` | Clear filled side from `_active_quotes` tracking |
| `bot_handle_cancel_response` | Clear cancelled side from `_active_quotes` tracking |

#### Quote logic (`_requote`)

For each quoted symbol (A and C):
```
half_spread = BASE_HALF_SPREAD + round(sigma_long × fv × VOL_SPREAD_FACTOR)
skew        = net_position × INVENTORY_SKEW_PER_SHARE
bid_px      = round(FV − half_spread − skew)
ask_px      = round(FV + half_spread − skew)
```

- Cancels existing bid/ask before placing fresh ones.
- Throttled: minimum 0.5s between requotes (configurable), unless FV moved > 1 tick (immediate).
- Respects `MAX_POSITION = 150` — reduces quote size near the limit.

#### ETF arbitrage (`_check_etf_arb`)

Checks every 0.5s:
```
FV_ETF = FV_A + B_mid + FV_C     (B_mid used as proxy since no B model)

If ETF_mid < FV_ETF − swap_cost − 3:  buy ETF at market + fromETF swap
If ETF_mid > FV_ETF + swap_cost + 3:  toETF swap + sell ETF at market
```

#### Background `trade()` loop

Runs every 0.5s: periodic requote for A and C + ETF arb check.

---

## Parameters to Calibrate

### From the competition brief (will be provided on the day)
| Parameter | Location | Description |
|---|---|---|
| `PE_0_A` | `models.py:217` | Baseline P/E for Stock A |
| `GAMMA_A` | `models.py:224` | Rate sensitivity of A's P/E |
| `PE_0_C` | `models.py:361` | Baseline P/E for Stock C |
| `GAMMA_C` | `models.py:364` | Rate sensitivity of C's P/E |
| `B_0` | `models.py:376` | Initial bond portfolio value |
| `D` | `models.py:377` | Duration |
| `CONV` | `models.py:378` | Convexity |
| `N` | `models.py:379` | Shares outstanding |
| `LAMBDA` | `models.py:380` | Bond PnL pass-through weight |

### From practice round observation
| Parameter | Location | How to tune |
|---|---|---|
| `Y_0` | `models.py:24` | Set so FV_C matches C's opening market price when E[Δr] = 0 |
| `BETA_Y` | `models.py:30` | Watch C move as pred markets shift; back out `Δy / E[Δr]` |
| `CPI_SENSITIVITY` | `models.py:39` | Watch R_HIKE/CUT repricing after a CPI print |
| `BASE_HALF_SPREAD` | `trading_bot_v1.py:76` | Match/undercut existing bot spreads on A and C |
| `VOL_SPREAD_FACTOR` | `trading_bot_v1.py:81` | Tune so spread widens ~2–5x during vol spikes |
| `INVENTORY_SKEW_PER_SHARE` | `trading_bot_v1.py:91` | Increase if ending rounds with large net position |

**Calibration check** — add this to the trade loop during the practice round:
```python
if self.stock_a.is_ready():
    print(f"A: FV={self.stock_a.fair_value:.1f}  mid={self.market_state.mid('A')}  diff={self.stock_a.fair_value - self.market_state.mid('A'):.1f}")
if self.stock_c.is_ready():
    print(f"C: FV={self.stock_c.fair_value:.1f}  mid={self.market_state.mid('C')}  diff={self.stock_c.fair_value - self.market_state.mid('C'):.1f}")
```
- FV tracks market mid closely → well calibrated
- Consistently off by fixed amount → wrong `PE_0`
- Diverges over time → wrong `GAMMA` or `BETA_Y`

---

## What's Not Yet Implemented

### Options on Stock B
`B_C_950/1000/1050` and `B_P_950/1000/1050` — currently no logic.
Planned approach:
- Estimate implied vol from the option chain each tick
- Enforce put-call parity: if `C − P ≠ S − Ke^{-rT}`, trade the mispriced leg
- Box spread detection across two strikes

### Prediction market trading
We read `R_HIKE/HOLD/CUT` to compute yields but never post quotes or take positions there. Edge available when CPI surprise shifts our estimates before the market reprices.

### Unstructured news parsing
Headlines are logged but not acted on. At minimum: keyword matching for rate-related language to nudge `_cpi_adjustment`.

### Stale order cancellation
No mechanism to cancel limit orders that haven't filled after N ticks. Adverse selection risk grows the longer a stale order sits.

### ETF arb — acquire-then-swap path
The `toETF` arb leg (swap components → sell ETF) currently requires holding A+B+C already. Needs logic to first acquire components if inventory is low.
