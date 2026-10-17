"""
Strategy 6: Liquidity Sweep + Break of Structure (BOS)
Educational backtest only. No live orders are placed.

Data source: Binance public OHLCV via ccxt.
Default execution timeframe: 5m; higher-timeframe trend filters: 15m and 1h.
"""

import os
import sys
import time
import math
import traceback
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd


SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
DAYS = int(os.getenv("DAYS", "90"))
INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", "1000"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.5")) / 100.0
RR = float(os.getenv("RR", "2.0"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))  # 0.10% per side; adjust to your exchange
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))  # 0.05% per side
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "12"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "96"))
MIN_BARS = 500


def timeframe_minutes(tf):
    units = {"m": 1, "h": 60, "d": 1440}
    n, unit = int(tf[:-1]), tf[-1]
    if unit not in units:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return n * units[unit]


def fetch_ohlcv(exchange, symbol, timeframe, since_ms, limit=1000):
    """Fetch history in pages, respecting exchange limits."""
    rows = []
    cursor = since_ms
    tf_ms = timeframe_minutes(timeframe) * 60_000
    now_ms = exchange.milliseconds()
    while cursor < now_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        except Exception as exc:
            print(f"Fetch warning: {exc}; retrying in 2 seconds...", flush=True)
            time.sleep(2)
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 2:
            break
        time.sleep(exchange.rateLimit / 1000)
    if not rows:
        raise RuntimeError("No OHLCV data returned by the exchange.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    # Exclude the currently forming candle.
    candle_ms = timeframe_minutes(timeframe) * 60_000
    latest_open_ms = int(df.index[-1].timestamp() * 1000)
    if latest_open_ms + candle_ms > exchange.milliseconds():
        df = df.iloc[:-1]
    return df.astype(float)


def resample_ohlcv(df, rule):
    out = df.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    return out


def add_indicators(df):
    x = df.copy()
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    x["atr"] = tr.rolling(ATR_PERIOD).mean()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    return x


def add_htf_trend(execution_df, exchange, symbol, since_ms):
    """Merge only completed 15m/1h candle trend values onto execution candles."""
    base_tf = TIMEFRAME
    base_minutes = timeframe_minutes(base_tf)
    result = execution_df.copy()

    for tf in ("15m", "1h"):
        if timeframe_minutes(tf) <= base_minutes:
            # If execution timeframe is equal to or above this TF, this filter is omitted.
            result[f"{tf}_bull"] = True
            result[f"{tf}_bear"] = True
            continue
        higher = fetch_ohlcv(exchange, symbol, tf, since_ms)
        higher = add_indicators(higher)
        # Shift by one full higher-timeframe candle to avoid using a candle before it closes.
        higher["bull"] = ((higher["close"] > higher["ema50"]) & (higher["ema20"] > higher["ema50"])).shift(1)
        higher["bear"] = ((higher["close"] < higher["ema50"]) & (higher["ema20"] < higher["ema50"])).shift(1)
        h = higher[["bull", "bear"]].reset_index().sort_values("timestamp")
        left = result.reset_index().sort_values("timestamp")
        merged = pd.merge_asof(left, h, on="timestamp", direction="backward")
        result[f"{tf}_bull"] = merged["bull"].fillna(False).to_numpy()
        result[f"{tf}_bear"] = merged["bear"].fillna(False).to_numpy()
    return result


def make_signals(df):
    x = add_indicators(df)
    # Prior swing levels only: shifted rolling windows prevent look-ahead.
    x["prior_high"] = x["high"].rolling(SWING_LOOKBACK).max().shift(1)
    x["prior_low"] = x["low"].rolling(SWING_LOOKBACK).min().shift(1)

    # Liquidity sweep: pierce a prior level, then close back inside the range.
    x["sweep_low"] = (x["low"] < x["prior_low"]) & (x["close"] > x["prior_low"])
    x["sweep_high"] = (x["high"] > x["prior_high"]) & (x["close"] < x["prior_high"])

    # Remember the most recent sweep extreme for a limited number of bars.
    x["long_sweep_extreme"] = x["low"].where(x["sweep_low"]).ffill()
    x["short_sweep_extreme"] = x["high"].where(x["sweep_high"]).ffill()
    x["bars_since_long_sweep"] = np.where(x["sweep_low"], 0, np.nan)
    x["bars_since_long_sweep"] = x["bars_since_long_sweep"].ffill()
    x["bars_since_long_sweep"] = x.groupby(x["sweep_low"].cumsum()).cumcount()
    x["bars_since_short_sweep"] = np.where(x["sweep_high"], 0, np.nan)
    x["bars_since_short_sweep"] = x["bars_since_short_sweep"].ffill()
    x["bars_since_short_sweep"] = x.groupby(x["sweep_high"].cumsum()).cumcount()

    # BOS: close crosses the recent prior range boundary after a sweep.
    x["bos_up"] = (x["close"] > x["prior_high"]) & (x["close"].shift(1) <= x["prior_high"].shift(1))
    x["bos_down"] = (x["close"] < x["prior_low"]) & (x["close"].shift(1) >= x["prior_low"].shift(1))
    sweep_window = max(3, min(12, SWING_LOOKBACK))
    long_ok = (
        x["bos_up"] &
        (x["bars_since_long_sweep"] <= sweep_window) &
        (x["close"] > x["ema20"]) &
        x["15m_bull"].astype(bool) &
        x["1h_bull"].astype(bool)
    )
    short_ok = (
        x["bos_down"] &
        (x["bars_since_short_sweep"] <= sweep_window) &
        (x["close"] < x["ema20"]) &
        x["15m_bear"].astype(bool) &
        x["1h_bear"].astype(bool)
    )
    x["long_signal"] = long_ok.fillna(False)
    x["short_signal"] = short_ok.fillna(False)
    return x


def backtest(df):
    equity = INITIAL_EQUITY
    peak_equity = equity
    max_dd = 0.0
    trades = []
    position = None
    rows = df.reset_index()

    for i in range(1, len(rows) - 1):
        bar = rows.iloc[i]
        next_bar = rows.iloc[i + 1]
        ts = bar["timestamp"]

        if position is not None:
            side = position["side"]
            exit_price = None
            reason = None

            # Conservative assumption: if SL and TP are touched in the same candle, SL is hit first.
            if side == "long":
                if bar["low"] <= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                elif bar["high"] >= position["tp"]:
                    exit_price, reason = position["tp"], "TP"
            else:
                if bar["high"] >= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                elif bar["low"] <= position["tp"]:
                    exit_price, reason = position["tp"], "TP"

            if exit_price is None and i - position["entry_i"] >= MAX_HOLD_BARS:
                exit_price, reason = bar["close"], "TIME"
            if exit_price is not None:
                # Apply adverse slippage on exits; entry slippage was applied at entry.
                if side == "long":
                    actual_exit = exit_price * (1 - SLIPPAGE_RATE)
                    gross = (actual_exit - position["entry"]) * position["qty"]
                else:
                    actual_exit = exit_price * (1 + SLIPPAGE_RATE)
                    gross = (position["entry"] - actual_exit) * position["qty"]
                fees = (position["entry"] * position["qty"] + actual_exit * position["qty"]) * FEE_RATE
                net = gross - fees
                equity += net
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "side": side.upper(), "entry": position["entry"], "exit": actual_exit,
                    "qty": position["qty"], "gross_pnl": gross, "fees": fees,
                    "net_pnl": net, "return_pct_on_equity": 100 * net / position["equity_at_entry"],
                    "exit_reason": reason, "equity_after": equity
                })
                position = None
                peak_equity = max(peak_equity, equity)
                max_dd = max(max_dd, (peak_equity - equity) / peak_equity if peak_equity > 0 else 0)

        if position is None:
            signal_side = "long" if bool(bar.get("long_signal", False)) else (
                "short" if bool(bar.get("short_signal", False)) else None
            )
            if signal_side is None or not np.isfinite(bar.get("atr", np.nan)) or bar["atr"] <= 0:
                continue

            entry_raw = float(next_bar["open"])
            entry = entry_raw * (1 + SLIPPAGE_RATE if signal_side == "long" else 1 - SLIPPAGE_RATE)
            if signal_side == "long":
                sweep_extreme = bar.get("long_sweep_extreme", np.nan)
                # A valid stop must be below entry; ATR provides a fallback for distant/missing extremes.
                sl = float(sweep_extreme) if np.isfinite(sweep_extreme) and sweep_extreme < entry else entry - 1.5 * bar["atr"]
                risk_per_unit = entry - sl
                tp = entry + RR * risk_per_unit
            else:
                sweep_extreme = bar.get("short_sweep_extreme", np.nan)
                sl = float(sweep_extreme) if np.isfinite(sweep_extreme) and sweep_extreme > entry else entry + 1.5 * bar["atr"]
                risk_per_unit = sl - entry
                tp = entry - RR * risk_per_unit

            if risk_per_unit <= 0 or not np.isfinite(risk_per_unit):
                continue
            risk_cash = max(0.0, equity) * RISK_PCT
            qty = risk_cash / risk_per_unit
            # Spot-like notional cap to prevent unlimited leverage in the simulation.
            qty = min(qty, max(0.0, equity) / entry)
            if qty <= 0:
                continue
            position = {
                "side": signal_side, "entry": entry, "sl": sl, "tp": tp,
                "qty": qty, "entry_i": i + 1,
                "entry_time": next_bar["timestamp"], "equity_at_entry": equity
            }

    # Close any remaining open trade at the last close.
    if position is not None:
        last = rows.iloc[-1]
        raw_exit = float(last["close"])
        actual_exit = raw_exit * (1 - SLIPPAGE_RATE if position["side"] == "long" else 1 + SLIPPAGE_RATE)
        gross = ((actual_exit - position["entry"]) if position["side"] == "long"
                 else (position["entry"] - actual_exit)) * position["qty"]
        fees = (position["entry"] * position["qty"] + actual_exit * position["qty"]) * FEE_RATE
        net = gross - fees
        equity += net
        trades.append({
            "entry_time": position["entry_time"], "exit_time": last["timestamp"],
            "side": position["side"].upper(), "entry": position["entry"], "exit": actual_exit,
            "qty": position["qty"], "gross_pnl": gross, "fees": fees,
            "net_pnl": net, "return_pct_on_equity": 100 * net / position["equity_at_entry"],
            "exit_reason": "END", "equity_after": equity
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = {
            "symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": len(df), "trades": 0,
            "initial_equity": INITIAL_EQUITY, "final_equity": equity,
            "net_pnl": equity - INITIAL_EQUITY, "return_pct": 100 * (equity / INITIAL_EQUITY - 1),
            "win_rate_pct": 0, "profit_factor": 0, "max_drawdown_pct": 0,
            "message": "No trades generated. Try a longer period or another liquid symbol."
        }
    else:
        wins = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"].sum()
        losses = -trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"].sum()
        pf = wins / losses if losses > 0 else (float("inf") if wins > 0 else 0)
        summary = {
            "symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": len(df), "trades": len(trades_df),
            "initial_equity": INITIAL_EQUITY, "final_equity": equity,
            "net_pnl": equity - INITIAL_EQUITY, "return_pct": 100 * (equity / INITIAL_EQUITY - 1),
            "win_rate_pct": 100 * (trades_df["net_pnl"] > 0).mean(),
            "profit_factor": pf, "max_drawdown_pct": 100 * max_dd,
            "avg_trade_net": trades_df["net_pnl"].mean(),
            "total_fees": trades_df["fees"].sum(),
            "tp_exits": int((trades_df["exit_reason"] == "TP").sum()),
            "sl_exits": int((trades_df["exit_reason"] == "SL").sum()),
        }
    return summary, trades_df


def main():
    if TIMEFRAME not in ("5m", "15m", "1h"):
        raise ValueError("TIMEFRAME must be one of: 5m, 15m, 1h")
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    since_ms = exchange.milliseconds() - DAYS * 24 * 60 * 60 * 1000
    print(f"Downloading {SYMBOL} {TIMEFRAME} data for {DAYS} days...", flush=True)
    df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, since_ms)
    if len(df) < MIN_BARS:
        raise RuntimeError(f"Only {len(df)} candles received; need at least {MIN_BARS}. Increase DAYS.")
    df = add_htf_trend(df, exchange, SYMBOL, since_ms)
    df = make_signals(df)
    summary, trades = backtest(df)

    os.makedirs("results", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trades_path = f"results/trades_{SYMBOL.replace('/', '_')}_{TIMEFRAME}_{stamp}.csv"
    summary_path = f"results/summary_{SYMBOL.replace('/', '_')}_{TIMEFRAME}_{stamp}.txt"
    trades.to_csv(trades_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("\n========== BACKTEST SUMMARY ==========")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"\nTrades CSV saved to: {trades_path}")
    print(f"Summary saved to: {summary_path}")
    print("Reminder: historical backtest results do not guarantee future performance.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"BACKTEST FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
