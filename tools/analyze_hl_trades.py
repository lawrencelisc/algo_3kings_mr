#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze `hl_trade_record.csv` (or path via argv) for R:R/ATR sensitivity and risk metrics.

Expected columns (case-insensitive, flexible):
  - pnl, net_pnl, profit, pnl_usd: per-trade PnL
  - time, timestamp, ts, datetime: time index
  - r_multiple, rr, r_r (optional) for R:R gate analysis
  - atr, atr_entry_mult (optional) for ATR guard analysis
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _ann_factor_from_times(ts: pd.Series) -> float:
    if len(ts) < 3:
        return math.sqrt(252.0)  # daily default
    dt = ts.diff().dt.total_seconds().median() or 3600.0
    if dt <= 0:
        dt = 3600.0
    n_per_year = (365.25 * 24.0 * 3600.0) / float(dt)
    return math.sqrt(max(n_per_year, 1.0))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd)) if len(dd) else 0.0


def sortino(returns: np.ndarray, ann_factor: float, tbf: float = 0.0) -> float:
    d = returns - tbf
    neg = d.copy()
    neg[neg > 0] = 0.0
    downside = float(np.sqrt((neg**2).mean() + 1e-18)) or 1e-12
    return float((d.mean() / downside) * ann_factor)


def analyze(df: pd.DataFrame) -> Tuple[float, float, float, str]:
    c = {str(x).lower().strip(): x for x in df.columns}
    pcol = None
    for k in ("pnl", "net_pnl", "profit", "pnl_usd", "realized_pnl"):
        if k in c:
            pcol = c[k]
            break
    if pcol is None:
        raise ValueError("Need a pnl column (pnl, net_pnl, profit, …)")
    pnl = df[pcol].astype(float)
    tcol = None
    for k in ("timestamp", "time", "ts", "datetime", "date"):
        if k in c:
            tcol = c[k]
            break
    if tcol is not None:
        t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    else:
        t = pd.RangeIndex(0, len(df))
    r = pnl.cumsum()
    eq = 600.0 + r  # 與紙上預設初始資金一致；可改為實盤權益起點再分析 MDD
    mdd = max_drawdown(eq.values.astype(float))
    rets = pnl.values.astype(float)
    ann = _ann_factor_from_times(t) if tcol and hasattr(t, "dt") else math.sqrt(252.0 * 24.0)
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * ann
    srt = sortino(rets, ann, 0.0)
    msg_parts = [f"Sharpe~{sharpe:.2f}, Sortino~{srt:.2f}, MaxDD~{mdd*100:.2f}% (point estimates from CSV)."]
    if "r_multiple" in c or "rr" in c or "r_r" in c:
        msg_parts.append("R:R columns present — slice by R:R>1.0 vs <0.8 in a notebook to test gate.")
    if "atr" in c or "atr_entry_mult" in c:
        msg_parts.append("ATR columns present — compare pnl of rows with atr>2*threshold vs <.")
    return float(sharpe), float(srt), float(mdd), " ".join(msg_parts)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "hl_trade_record.csv")
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        print("No CSV at", path, "— place hl_trade_record.csv and re-run.")
        print("*** Without history, 'expected' Sharpe/Sortino/MDD cannot be estimated from this repo.")
        return 1
    df = pd.read_csv(path)
    s, so, mdd, note = analyze(df)
    print(note)
    print(f"sharpe={s:.4f} sortino={so:.4f} max_drawdown_fraction={mdd:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
