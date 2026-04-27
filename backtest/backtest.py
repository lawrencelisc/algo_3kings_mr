#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest.py — 對 fetch_ohlcv.py 拉下來的 5m parquet 跑 backtest。

同時測試三條路徑：
  MR        — CHOP/NEUTRAL regime 均值回歸（現行核心）
  TREND_v1  — EMA9/21 cross（現行，已知滯後）
  TREND_v2  — ADX 連續上升 + EMA stack（改良版，減少滯後）

用法：
    python backtest.py [--data-dir ./data] [--out bt_result.csv]

輸出：
  bt_result.csv   每筆模擬交易明細
  bt_summary.csv  per-symbol × mode 摘要
  控制台          aggregate 對比表
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core.indicators import adx_series, ema_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backtest")


# ═══════════════════════════════════════════════════════════════════════════
# 參數
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BTConfig:
    equity_usdt: float      = 600.0
    risk_fraction: float    = 0.02
    leverage: float         = 8.0
    maker_rebate: float     = 0.00002
    round_trip_fee: float   = 0.0003

    # MR 信號（1H bars）
    z_window: int           = 45       # 45 × 1H = 45H 均值窗口
    z_entry_min: float      = 1.5
    z_entry_max: float      = 2.5
    min_tp_fee_multiple: float = 5.0

    # Regime（直接在 1H bars 上算，不需要 resample）
    adx_period: int         = 14
    adx_trend: float        = 25.0
    adx_chop: float         = 20.0
    adx_confirm_bars: int   = 3

    # ATR（14 根 1H）
    atr_period: int         = 14

    # TP/SL 倍數
    mr_sl_mult: float       = 1.0
    mr_tp_mult: float       = 0.5
    trend_tp_mult: float    = 2.0
    trend_sl_mult: float    = 1.0

    # 持倉限制（1H bars）
    max_hold_bars: int      = 4        # 4H 強制平倉
    sl_cooldown_bars: int   = 1        # 1H 冷卻

    bars_per_year: int      = 24 * 365


# ═══════════════════════════════════════════════════════════════════════════
# 指標計算
# ═══════════════════════════════════════════════════════════════════════════

def _atr_arr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    trs = np.zeros(n)
    for i in range(1, n):
        trs[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i]  - closes[i-1]))
    atr = np.full(n, np.nan)
    for i in range(period, n):
        atr[i] = trs[i - period + 1: i + 1].mean()
    return atr


def _z_arr(closes: np.ndarray, window: int) -> np.ndarray:
    n = len(closes)
    z = np.full(n, np.nan)
    for i in range(window, n):
        sl = closes[i - window: i]
        mu = sl.mean()
        sd = sl.std(ddof=1) or 1e-12
        z[i] = (closes[i] - mu) / sd
    return z


def compute_indicators(df: pd.DataFrame, cfg: BTConfig) -> pd.DataFrame:
    """1H 數據直接計算，不需要 resample。"""
    df = df.copy()
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)

    df["z"]   = _z_arr(c, cfg.z_window)
    df["atr"] = _atr_arr(h, l, c, cfg.atr_period)

    # ADX 直接在 1H 上算
    ohlcv = [[0, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)]
              for r in df.itertuples()]
    raw_adx = adx_series(ohlcv, cfg.adx_period)
    adx_col = np.full(len(df), np.nan)
    if raw_adx:
        adx_col[-len(raw_adx):] = raw_adx
    df["adx_1h"] = adx_col

    # EMA9 / EMA21
    e9v  = ema_series(c.tolist(), 9)
    e21v = ema_series(c.tolist(), 21)
    ema9  = np.full(len(df), np.nan)
    ema21 = np.full(len(df), np.nan)
    if e9v:  ema9[-len(e9v):]   = e9v
    if e21v: ema21[-len(e21v):] = e21v
    df["ema9_1h"]  = ema9
    df["ema21_1h"] = ema21

    # ADX 連續上升（過去 3 bar 單調遞增）
    adx_rising = np.zeros(len(df), dtype=bool)
    for i in range(2, len(adx_col)):
        if not any(np.isnan(adx_col[i-2:i+1])):
            adx_rising[i] = bool(adx_col[i] > adx_col[i-1] > adx_col[i-2])
    df["adx_rising"] = adx_rising

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 模擬
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    symbol: str
    mode: str
    side: str
    entry_i: int
    entry_price: float
    tp_price: float
    sl_price: float
    exit_i: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    net_pnl: float = 0.0
    hold_bars: int = 0


def _tp_sl_prices(entry: float, side: str, atr: float, mode: str, cfg: BTConfig,
                  closes_window: Optional[np.ndarray] = None) -> Tuple[float, float]:
    if mode == "MR" and closes_window is not None:
        mean45 = closes_window.mean()
        sl = atr * cfg.mr_sl_mult
        tp_atr = atr * cfg.mr_tp_mult
        if side == "long":
            tp = max(mean45, entry + tp_atr)
            return tp, entry - sl
        else:
            tp = min(mean45, entry - tp_atr)
            return tp, entry + sl
    elif mode in ("TREND_v1", "TREND_v2"):
        if side == "long":
            return entry + atr * cfg.trend_tp_mult, entry - atr * cfg.trend_sl_mult
        else:
            return entry - atr * cfg.trend_tp_mult, entry + atr * cfg.trend_sl_mult
    else:
        if side == "long":
            return entry + atr * cfg.mr_tp_mult, entry - atr * cfg.mr_sl_mult
        else:
            return entry - atr * cfg.mr_tp_mult, entry + atr * cfg.mr_sl_mult


def simulate(df: pd.DataFrame, symbol: str, cfg: BTConfig, mode: str) -> List[Trade]:
    trades: List[Trade] = []
    pos: Optional[Trade] = None
    cooldown = 0
    notional = cfg.equity_usdt * cfg.risk_fraction * cfg.leverage

    closes  = df["close"].values.astype(float)
    highs   = df["high"].values.astype(float)
    lows    = df["low"].values.astype(float)
    z_arr   = df["z"].values
    atr_arr = df["atr"].values
    adx_arr = df["adx_1h"].values
    e9_arr  = df["ema9_1h"].values
    e21_arr = df["ema21_1h"].values
    rising  = df["adx_rising"].values
    n = len(closes)
    warmup = max(cfg.atr_period, cfg.z_window) + 1

    for i in range(warmup, n):
        mark   = closes[i]
        atr_v  = atr_arr[i]
        adx_v  = adx_arr[i]
        e9_v   = e9_arr[i]
        e21_v  = e21_arr[i]
        z_v    = z_arr[i]
        rise_v = rising[i]

        if any(np.isnan(x) for x in [atr_v, adx_v, e9_v, e21_v, z_v]):
            continue
        if atr_v <= 0:
            continue

        # ── 平倉 ────────────────────────────────────────────────────────
        if pos is not None:
            hi, lo = highs[i], lows[i]
            reason = None
            exit_px = mark
            if pos.side == "long":
                if hi >= pos.tp_price: reason, exit_px = "TP", pos.tp_price
                elif lo <= pos.sl_price: reason, exit_px = "SL", pos.sl_price
            else:
                if lo <= pos.tp_price: reason, exit_px = "TP", pos.tp_price
                elif hi >= pos.sl_price: reason, exit_px = "SL", pos.sl_price
            if reason is None and (i - pos.entry_i) >= cfg.max_hold_bars:
                reason, exit_px = "TIMEOUT", mark
            if reason:
                amt = notional / pos.entry_price
                gross = ((exit_px - pos.entry_price) if pos.side == "long"
                         else (pos.entry_price - exit_px)) * amt
                net = gross + exit_px * amt * cfg.maker_rebate
                pos.exit_i, pos.exit_price = i, exit_px
                pos.exit_reason, pos.net_pnl = reason, net
                pos.hold_bars = i - pos.entry_i
                trades.append(pos)
                if reason == "SL": cooldown = cfg.sl_cooldown_bars
                pos = None
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        # ── 信號 ────────────────────────────────────────────────────────
        side = None

        if mode == "MR":
            if adx_v < cfg.adx_chop:
                if -cfg.z_entry_max <= z_v <= -cfg.z_entry_min:
                    side = "long"
                elif cfg.z_entry_min <= z_v <= cfg.z_entry_max:
                    side = "short"

        elif mode == "TREND_v1":
            if adx_v >= cfg.adx_trend:
                if e9_v > e21_v: side = "long"
                elif e9_v < e21_v: side = "short"

        elif mode == "TREND_v2":
            # ADX 連續上升（趨勢加速）+ EMA stack 方向
            if adx_v >= cfg.adx_trend and rise_v:
                if e9_v > e21_v: side = "long"
                elif e9_v < e21_v: side = "short"

        if side is None:
            continue

        cw = closes[i - cfg.z_window: i] if mode == "MR" else None
        tp, sl = _tp_sl_prices(mark, side, atr_v, mode, cfg, closes_window=cw)

        # 最小 TP 距離
        tp_dist = abs(tp - mark)
        sl_dist = abs(sl - mark)
        if tp_dist < mark * cfg.round_trip_fee * cfg.min_tp_fee_multiple:
            continue
        if sl_dist <= 0 or tp_dist / sl_dist < 0.8:
            continue

        pos = Trade(symbol=symbol, mode=mode, side=side,
                    entry_i=i, entry_price=mark, tp_price=tp, sl_price=sl)

    # EOD
    if pos is not None:
        exit_px = closes[-1]
        amt = notional / pos.entry_price
        gross = ((exit_px - pos.entry_price) if pos.side == "long"
                 else (pos.entry_price - exit_px)) * amt
        pos.exit_i, pos.exit_price = n-1, exit_px
        pos.exit_reason = "EOD"
        pos.net_pnl = gross + exit_px * amt * cfg.maker_rebate
        pos.hold_bars = n - 1 - pos.entry_i
        trades.append(pos)

    return trades


# ═══════════════════════════════════════════════════════════════════════════
# 統計
# ═══════════════════════════════════════════════════════════════════════════

def stats(trades: List[Trade], mode: str) -> Dict:
    t = [x for x in trades if x.mode == mode]
    if not t:
        return {"mode": mode, "n": 0, "total_pnl": 0, "sharpe": 0,
                "mdd": 0, "win_rate": 0, "avg_pnl": 0,
                "tp_rate": 0, "sl_rate": 0, "avg_hold": 0}
    pnls = np.array([x.net_pnl for x in t])
    wins = (pnls > 0).sum()
    eq = np.concatenate([[600.0], 600.0 + np.cumsum(pnls)])
    peak = np.maximum.accumulate(eq)
    mdd = float(((peak - eq) / np.where(peak>0,peak,1)).max()) * 100
    ann = math.sqrt(252 * 24 * 12)
    sh = float((pnls.mean() / (pnls.std() + 1e-12)) * ann)
    return {
        "mode":      mode,
        "n":         len(t),
        "win_rate":  round(wins / len(t) * 100, 1),
        "total_pnl": round(pnls.sum(), 4),
        "avg_pnl":   round(pnls.mean(), 4),
        "sharpe":    round(sh, 3),
        "mdd":       round(mdd, 2),
        "tp_rate":   round(sum(1 for x in t if x.exit_reason=="TP")/len(t)*100, 1),
        "sl_rate":   round(sum(1 for x in t if x.exit_reason=="SL")/len(t)*100, 1),
        "avg_hold":  round(sum(x.hold_bars for x in t)/len(t), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out",      default="bt_result.csv")
    ap.add_argument("--summary",  default="bt_summary.csv")
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    if not files:
        logger.error("找不到 parquet，請先跑 fetch_ohlcv.py")
        return

    cfg = BTConfig()
    all_trades: List[Trade] = []
    rows_summary = []

    for f in files:
        symbol = f.stem.replace("_", "/", 1).replace("_", ":", 1)
        logger.info("%-25s ...", symbol)
        try:
            df = pd.read_parquet(f)
            df = compute_indicators(df, cfg)
        except Exception as e:
            logger.warning("  skip %s: %s", symbol, e)
            continue

        for mode in ("MR", "TREND_v1", "TREND_v2"):
            t = simulate(df, symbol, cfg, mode)
            all_trades.extend(t)
            s = stats(t, mode)
            s["symbol"] = symbol
            rows_summary.append(s)
            logger.info("  %-12s n=%-4d win=%-5s pnl=%+.2f sharpe=%.2f mdd=%.1f%%",
                        mode, s["n"], f"{s['win_rate']:.0f}%",
                        s["total_pnl"], s["sharpe"], s["mdd"])

    # 輸出明細
    if all_trades:
        pd.DataFrame([{
            "symbol": x.symbol, "mode": x.mode, "side": x.side,
            "entry_price": x.entry_price, "exit_price": x.exit_price,
            "tp_price": x.tp_price, "sl_price": x.sl_price,
            "net_pnl": round(x.net_pnl, 6), "hold_bars": x.hold_bars,
            "exit_reason": x.exit_reason,
        } for x in all_trades]).to_csv(args.out, index=False)
        logger.info("明細 → %s", args.out)

    if rows_summary:
        pd.DataFrame(rows_summary).to_csv(args.summary, index=False)
        logger.info("Summary → %s", args.summary)

    # ── Aggregate 對比 ────────────────────────────────────────────────────
    W = 68
    print("\n" + "═"*W)
    print("  AGGREGATE RESULTS  (all symbols, 6 months, 5m bars)")
    print("═"*W)
    for mode in ("MR", "TREND_v1", "TREND_v2"):
        s = stats(all_trades, mode)
        if s["n"] == 0:
            print(f"  {mode:<12}  no trades")
            continue
        print(f"  {mode:<12}  "
              f"n={s['n']:<5}  win={s['win_rate']:4.0f}%  "
              f"pnl={s['total_pnl']:+8.2f}U  avg={s['avg_pnl']:+.4f}U  "
              f"sharpe={s['sharpe']:5.2f}  mdd={s['mdd']:5.1f}%  "
              f"tp={s['tp_rate']:.0f}%  sl={s['sl_rate']:.0f}%  "
              f"hold={s['avg_hold']:.0f}bar")
    print("═"*W)
    print()
    print("  TREND_v1 = EMA9/21 cross         (現行，已知滯後)")
    print("  TREND_v2 = ADX 連升 + EMA stack  (改良，動能確認)")
    print()
    print("  決策標準：")
    print("  - TREND_v2 sharpe < MR sharpe → 停掉 TREND 路徑，只跑 MR")
    print("  - TREND_v2 sharpe > TREND_v1   → 換掉 EMA cross 用 ADX rising")
    print("═"*W)


if __name__ == "__main__":
    main()
