# Trading Bot — Design & Reasoning

## Overview

The bot is structured as three layers:

```
trading_bot.py   — event handlers, order execution, risk gating
models.py        — fair value computation (Stocks A/C, ETF, yield model)
state.py         — raw market state: book metrics, trade history, volatility
```

The exchange client (`XChangeClient`) handles the gRPC transport and maintains raw order books and positions. Our code sits on top of it.

---

## Phase 1 — Market State (`state.py`)

### Responsibility

Track all derived metrics that downstream logic (fair value models, signal generation, risk) needs, without those modules having to touch the raw order books directly.

### Data Model

#### `SymbolState` (one per symbol)

| Field | Source | Formula |
|---|---|---|
| `mid` | book update | `(best_bid + best_ask) / 2` |
| `weighted_mid` | book update | `(best_bid × ask_size + best_ask × bid_size) / (bid_size + ask_size)` |
| `imbalance` | book update | `(bid_size − ask_size) / (bid_size + ask_size)` |
| `last_trade` | trade message | raw exchange price |
| `sigma_short` | trade history | sample std of last 20 returns |
| `sigma_long` | trade history | sample std of last 100 returns |
| `vol_ratio` | computed | `sigma_short / sigma_long` |

#### `MarketState`

Top-level container. The bot holds one instance and calls:
- `on_book_update(symbol, bids, asks)` — from `bot_handle_book_update`
- `on_trade(symbol, price)` — from `bot_handle_trade_msg`

---

### Design Decisions

#### Why `weighted_mid` instead of plain `mid`?

Plain mid is equally weighted between bid and ask regardless of how much size sits at each level. Weighted mid accounts for resting liquidity:

- Large ask size, small bid size → weighted mid is pulled *below* simple mid → signal that sellers dominate → more likely to trade down.
- This gives a better short-term price forecast than raw mid, especially around news events when one side gets hit hard.

#### Why `imbalance`?

Imbalance is a well-documented short-horizon predictor of price direction (Cont, Kukanov & Stoikov, 2014). A strongly positive imbalance (+0.6 to +1.0) means bid depth dominates, suggesting upward pressure — we'd be more willing to lift offers. We use it as a signal modifier rather than a standalone trigger.

#### Return computation: simple returns vs log returns

We use simple returns `r = (P_t − P_{t-1}) / P_{t-1}` rather than log returns `ln(P_t / P_{t-1})`. For tick-level data with small moves these are nearly identical. Simple returns are slightly faster to compute and easier to reason about when the numbers go into threshold comparisons.

#### Window sizes: 20 / 100 ticks

Time structure: 5 ticks/second × 90 seconds/day = 450 ticks/day.

- **SHORT_WINDOW = 20 ticks ≈ 4 seconds** — captures recent micro-volatility. Sensitive enough to react to a news spike within a few seconds.
- **LONG_WINDOW = 100 ticks ≈ 20 seconds** — provides a stable baseline that doesn't drift too fast but still adapts within a single day.

These are starting values and should be tuned once we can replay historical data.

#### `vol_ratio` as a regime indicator

`vol_ratio = sigma_short / sigma_long`.

- ≈ 1.0 → normal conditions, proceed with standard quote sizes.
- > 1.5 → short-term vol elevated (news hit, large order flow). Tighten position limits, widen quotes, or pause new entries.
- < 0.7 → unusually quiet. May indicate a stale book (no recent trades). Treat metrics with skepticism.

#### Rolling windows via `deque(maxlen=N)`

Using `deque` with a fixed `maxlen` is O(1) append and O(1) drop of the oldest element — no shifting, no manual pruning. The alternative (a plain list with `list.pop(0)`) is O(N) per drop. At 5 ticks/second this doesn't matter much for N=100, but the deque approach is also conceptually cleaner.

We store `LONG_WINDOW + 1` prices (one extra) so we can always compute a return from the second-to-last price without maintaining a separate `prev_price` variable.

#### Pure Python `_std` instead of numpy

numpy's `std` has measurable import and first-call overhead. For windows of 20–100 elements, the pure Python implementation is fast enough (< 10 µs per call on modern hardware). If profiling shows this is a bottleneck we can swap to `numpy.std` without changing the interface.

#### `None` sentinel vs 0.0

All metrics default to `None` rather than `0.0`. This forces callers to explicitly guard before using a value in trading logic (`if state.mid("A") is not None`). A `0.0` mid is a valid price on some exchanges but would be catastrophically wrong as a fair value input — `None` makes the "not yet computed" state explicit and hard to ignore.

---

## Phase 2 — Rate & Yield Model (`models.py`)

### `YieldModel`

Derives `y_t` from two update sources:

| Source | Trigger | Effect |
|---|---|---|
| Prediction market book tick | `R_HIKE / R_HOLD / R_CUT` book update | Read mids → normalise → recompute `y_t` |
| CPI structured news | `cpi_print` news event | Apply surprise adjustment → recompute `y_t` |

#### Prediction market → probabilities

Prediction market prices are in [0, 100] (cents on the dollar for a $1 payout). A mid of 35 on `R_HIKE` means ~35% implied probability of a hike. We divide by 100, then normalise the three values to sum exactly to 1.0 — the raw mids drift off because of bid-ask spread and stale quotes.

#### Expected rate change

```
E[Δr] = 25 × q_hike + 0 × q_hold − 25 × q_cut   (basis points)
```

#### Yield

```
y_t = y_0 + β_y × E[Δr]
delta_y = y_t − y_0
```

`delta_y` is passed directly to the bond portfolio model in Phase 4.

#### CPI surprise adjustment

```
surprise = actual − forecast
adjustment = clamp(CPI_SENSITIVITY × surprise, −0.3, +0.3)
```

The adjustment is stored as `_cpi_adjustment` — a persistent offset added to `q_hike` and subtracted from `q_cut` on every subsequent market update. This persists across market ticks so the CPI signal isn't immediately washed out by the next book snapshot. Each new CPI print overwrites the previous adjustment.

**Why front-run the prediction market?** The prediction market reprices slowly (it requires other participants to trade). Our CPI adjustment immediately shifts our fair value estimate before the market catches up, giving us an edge in the first few ticks after a print.

#### Parameters to calibrate

| Parameter | Default | Meaning |
|---|---|---|
| `Y_0` | 0.045 | Baseline yield (set to competition start rate) |
| `BETA_Y` | 0.0001 | Yield sensitivity to expected rate change |
| `CPI_SENSITIVITY` | 0.1 | Prob shift per unit of CPI surprise |

---

## Phase 3 — Stock A Fair Value (`StockAModel`)

### Formula

```
PE_t  = PE_0 × exp(−γ × delta_y)
FV_A  = EPS_A × PE_t
```

### Two update triggers

| Trigger | Source | Action |
|---|---|---|
| `on_earnings(eps)` | Structured news, asset="A" | Store EPS, recompute FV_A |
| `on_yield_change()` | After any `YieldModel` update | Recompute PE_t and FV_A with latest delta_y |

Reacting to both triggers means FV_A is always current. A yield spike between two earnings events still gets reflected immediately — we don't wait for the next EPS print.

### Why the exponential P/E form?

```
PE_t = PE_0 × exp(−γ × delta_y)
```

- At `delta_y = 0`: `PE_t = PE_0` (baseline; no rate change).
- `delta_y > 0` (rates up): exp < 1 → P/E compresses (future earnings worth less).
- `delta_y < 0` (rates down): exp > 1 → P/E expands.

A linear model (`PE_t = PE_0 − k × delta_y`) could go negative for large rate shocks. The exponential form is always positive and is standard in rate-sensitive equity valuation.

### Yield fallback

If `YieldModel` hasn't received a quote yet, `delta_y` defaults to 0 so `PE_t = PE_0`. This means an early earnings print still produces a usable FV estimate rather than returning None.

### `mispricing()` helper

Returns `FV_A − market_price`. The trading layer uses this directly:
- Large positive → buy aggressively
- Small positive/negative → quote passively around FV
- Large negative → sell

### Parameters to calibrate

| Parameter | Default | Meaning |
|---|---|---|
| `PE_0_A` | 15.0 | Baseline P/E at reference yield |
| `GAMMA_A` | 10.0 | Rate sensitivity of P/E multiple |

---

## Phase 4 — Stock C Fair Value (`StockCModel`)

### Formula

```
PE_t             = PE_0 × exp(−γ × delta_y)
operations_value = EPS_C × PE_t
ΔB               = B_0 × (−D × Δy  +  0.5 × Conv × Δy²)
bond_impact      = λ × (ΔB / N)
FV_C             = operations_value + bond_impact
```

### Two update triggers

Same pattern as `StockAModel`:

| Trigger | Source | Action |
|---|---|---|
| `on_earnings(eps)` | Structured news, asset="C" | Store EPS, recompute FV_C |
| `on_yield_change()` | After any `YieldModel` update | Recompute PE_t, ΔB, and FV_C |

### Bond portfolio model — why the Taylor expansion?

The exact bond price-yield relationship is non-linear. The duration-convexity approximation is a second-order Taylor expansion around `y_0`:

```
ΔB ≈ B_0 × (−D × Δy  +  0.5 × Conv × Δy²)
```

- **First-order term** `−D × Δy`: linear rate sensitivity. Yields up → bond value down. Duration `D` is the primary lever.
- **Second-order term** `0.5 × Conv × Δy²`: always positive (Δy² ≥ 0). This is convexity — bonds gain more than duration predicts when yields fall, and lose less when yields rise. For large Δy this correction is significant.

### Why Stock C is more rate-sensitive than Stock A

Stock A's only rate channel is the P/E compression/expansion. Stock C has two:
1. P/E channel (same as A)
2. Direct bond portfolio repricing (ΔB)

A yield spike hits both simultaneously, making FV_C potentially very sensitive. LAMBDA and the bond parameters need careful calibration — miscalibrated bond parameters produce large, noisy FV_C estimates.

### Decomposed components

`operations_value` and `bond_impact` are stored as separate attributes so trading logic can inspect which component is driving a mispricing — useful for deciding whether to hedge with prediction markets (rate-driven move) or wait for an earnings catalyst.

`delta_b` is computed even before the first EPS print arrives, so we can track bond sensitivity from tick one.

### Parameters to calibrate

| Parameter | Default | Meaning |
|---|---|---|
| `PE_0_C` | 12.0 | Baseline P/E at reference yield |
| `GAMMA_C` | 10.0 | Rate sensitivity of P/E multiple |
| `B_0` | 1,000,000 | Initial bond portfolio value |
| `D` | 7.0 | Modified duration (years) |
| `CONV` | 60.0 | Convexity |
| `N` | 10,000 | Shares outstanding |
| `LAMBDA` | 1.0 | Bond PnL pass-through to stock price |

---

## Phase 5 — ETF Arbitrage *(planned)*

- `FV_ETF = FV_A + FV_B + FV_C`
- If `market_ETF > FV_ETF + swap_cost(5)`: sell ETF, buy components via swap.
- If `market_ETF < FV_ETF − swap_cost(5)`: buy ETF, sell components via swap.
- Swap cost is flat 5 per swap regardless of quantity.

---

## Phase 6 — Options on Stock B *(planned)*

- No fundamental model. Use rolling vol (`sigma_long` for B) to price via Black-Scholes.
- Enforce no-arbitrage constraints: call/put bounds, monotonicity across strikes.
- Trade when market price violates a bound or deviates significantly from BS price.

---

## Risk Management *(planned)*

- Hard position limits per symbol.
- Inventory skew: if long, post more aggressively on ask side; if short, on bid side.
- Cancel stale limit orders (posted N ticks ago and not filled) to avoid adverse selection.
- Suspend trading when `vol_ratio > threshold` (news regime).
