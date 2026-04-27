#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wfa.py — Walk-Forward Analysis：反趨勢 MR（ADX rising 反向入場）參數優化。

策略邏輯
--------
ADX 連升 N 根 1H bar（趨勢加速）→ 視為過熱 → 反向 MR 入場：
  EMA9 > EMA21（上升趨勢）→ 做空（預期回調）
  EMA9 < EMA21（下降趨勢）→ 做多（預期反彈）

WFA 結構
--------
  全歷史切成 n_splits 個 fold：
    IS  (in-sample)  : 前 is_bars 根 1H bar — 網格搜索最佳參數
    OOS (out-of-sample): 後 oos_bars 根 1H bar — 用 IS 最佳參數驗證
  每個 fold 向前滾動 oos_bars（anchored = False）或固定起點（anchored = True）

優化目標：IS Sharpe（annualized）
報告指標：OOS Sharpe / MDD / win_rate / avg_pnl / n_trades

用法：
    python wfa.py [--data-dir ./data] [--out wfa_result.csv]

輸出：
    wfa_result.csv    每個 fold × symbol × param_set 的 OOS 結果
    wfa_best.csv      每個 fold 的最佳參數 + OOS 成績
    控制台            aggregate OOS 統計
"""
from __future__ import annotations

import argparse
import itertools
import logging
import math
import sys
from dataclasses import dataclass
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
logger = logging.getLogger("wfa")


# ═══════════════════════════════════════════════════════════════════════════
# 參數網格（針對反趨勢 MR 優化的參數）
# ═══════════════════════════════════════════════════════════════════════════

PARAM_GRID = {
    # ADX 連升確認 bar 數（趨勢過熱判斷）
    "adx_rising_bars":  [2, 3, 4],

    # ADX 最低水平（確保是真趨勢，不是噪聲）
    "adx_min":          [22.0, 25.0, 28.0],

    # Z-score 入場門檻（反趨勢，不需要太極端）
    "z_entry_min":      [0.8, 1.0, 1.2, 1.5],
    "z_entry_max":      [2.0, 2.5, 3.0],

    # SL 倍數（反趨勢風險更大，SL 要給空間）
    "sl_mult":          [1.0, 1.5, 2.0],

    # 最大持倉時間（1H bars）
    "max_hold_bars":    [4, 6, 8],
}

# WFA 切分設定
IS_BARS  = 24 * 90    # 90 天 in-sample
OOS_BARS = 24 * 30    # 30 天 out-of-sample
MIN_TRADES_IS = 10    # IS 內最少交易筆數才算有效

# 固定參數
ADX_PERIOD   = 14
ATR_PERIOD   = 14
Z_WINDOW     = 45
MAKER_REBATE = 0.00002
ROUND_TRIP   = 0.000768   # 你的實際費率
EQUITY       = 600.0
RISK_FRAC    = 0.02
LEVERAGE     = 8.0

# MR symbols（backtest 確認有 MR edge）
MR_SYMBOLS = [
    "AAVE/USDC:USDC",
    "AXS/USDC:USDC",
    "FARTCOIN/USDC:USDC",
    "TRUMP/USDC:USDC",
    "HYPE/USDC:USDC",
    "XRP/USDC:USDC",
    "TAO/USDC:USDC",
]


# ═══════════════════════════════════════════════════════════════════════════
# 指標預計算
# ═══════════════════════════════════════════════════════════════════════════

def precompute(df: pd.DataFrame) -> pd.DataFrame:
    """預計算所有固定指標，避免在網格搜索中重複計算。"""
    df = df.copy()
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)

    # Z-score (window=45)
    z = np.full(n, np.nan)
    for i in range(Z_WINDOW, n):
        sl = c[i - Z_WINDOW: i]
        mu, sd = sl.mean(), sl.std(ddof=1) or 1e-12
        z[i] = (c[i] - mu) / sd
    df["z"] = z

    # ATR(14)
    trs = np.zeros(n)
    for i in range(1, n):
        trs[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = np.full(n, np.nan)
    for i in range(ATR_PERIOD, n):
        atr[i] = trs[i - ATR_PERIOD + 1: i + 1].mean()
    df["atr"] = atr

    # ADX(14)
    ohlcv = [[0, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)]
             for r in df.itertuples()]
    raw_adx = adx_series(ohlcv, ADX_PERIOD)
    adx_col = np.full(n, np.nan)
    if raw_adx:
        adx_col[-len(raw_adx):] = raw_adx
    df["adx"] = adx_col

    # EMA9 / EMA21
    e9v  = ema_series(c.tolist(), 9)
    e21v = ema_series(c.tolist(), 21)
    ema9  = np.full(n, np.nan)
    ema21 = np.full(n, np.nan)
    if e9v:  ema9[-len(e9v):]   = e9v
    if e21v: ema21[-len(e21v):] = e21v
    df["ema9"]  = ema9
    df["ema21"] = ema21

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 單次模擬（給定參數集）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    n: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    sharpe: float
    mdd: float
    tp_rate: float
    sl_rate: float
    avg_hold: float


def simulate_counter_trend(
    df: pd.DataFrame,
    adx_rising_bars: int,
    adx_min: float,
    z_entry_min: float,
    z_entry_max: float,
    sl_mult: float,
    max_hold_bars: int,
) -> SimResult:
    """
    反趨勢 MR 模擬：
    ADX 連升 adx_rising_bars 根且 ≥ adx_min → 觸發反向 MR 入場條件
    Z-score 在 [z_entry_min, z_entry_max] 方向確認入場
    TP = 45H 均值（Z 回歸目標）
    SL = entry ± sl_mult × ATR
    """
    closes  = df["close"].values.astype(float)
    highs   = df["high"].values.astype(float)
    lows    = df["low"].values.astype(float)
    z_arr   = df["z"].values
    atr_arr = df["atr"].values
    adx_arr = df["adx"].values
    e9_arr  = df["ema9"].values
    e21_arr = df["ema21"].values
    n = len(df)

    pnls: List[float] = []
    hold_bars_list: List[int] = []
    reasons: List[str] = []

    pos_side: Optional[str] = None
    pos_entry: float = 0.0
    pos_tp: float = 0.0
    pos_sl: float = 0.0
    pos_entry_i: int = 0
    cooldown: int = 0
    warmup = max(ATR_PERIOD, Z_WINDOW, ADX_PERIOD * 3) + adx_rising_bars + 2

    for i in range(warmup, n):
        mark   = closes[i]
        atr_v  = atr_arr[i]
        adx_v  = adx_arr[i]
        e9_v   = e9_arr[i]
        e21_v  = e21_arr[i]
        z_v    = z_arr[i]

        if any(np.isnan(x) for x in [atr_v, adx_v, e9_v, e21_v, z_v]):
            continue
        if atr_v <= 0:
            continue

        # ── 平倉 ────────────────────────────────────────────────────────
        if pos_side is not None:
            hi, lo = highs[i], lows[i]
            reason = None
            exit_px = mark

            if pos_side == "long":
                if hi >= pos_tp:  reason, exit_px = "TP", pos_tp
                elif lo <= pos_sl: reason, exit_px = "SL", pos_sl
            else:
                if lo <= pos_tp:  reason, exit_px = "TP", pos_tp
                elif hi >= pos_sl: reason, exit_px = "SL", pos_sl

            hold = i - pos_entry_i
            if hold >= max_hold_bars and reason is None:
                reason, exit_px = "TIMEOUT", mark

            if reason:
                amt = (EQUITY * RISK_FRAC * LEVERAGE) / pos_entry
                gross = ((exit_px - pos_entry) if pos_side == "long"
                         else (pos_entry - exit_px)) * amt
                net = gross + exit_px * amt * MAKER_REBATE
                pnls.append(net)
                hold_bars_list.append(hold)
                reasons.append(reason)
                if reason == "SL":
                    cooldown = 2
                pos_side = None
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        # ── ADX 連升判斷 ────────────────────────────────────────────────
        adx_window = adx_arr[i - adx_rising_bars + 1: i + 1]
        if any(np.isnan(adx_window)):
            continue
        # 連續上升 + 達到最低水平
        if not (adx_v >= adx_min):
            continue
        is_rising = all(adx_window[j] > adx_window[j-1]
                        for j in range(1, len(adx_window)))
        if not is_rising:
            continue

        # ── 反向 MR 信號 ────────────────────────────────────────────────
        # EMA 方向 → 反向入場
        # 上升趨勢（e9>e21）過熱 → Z 偏高 → 做空
        # 下降趨勢（e9<e21）過熱 → Z 偏低 → 做多
        side = None
        if e9_v > e21_v and z_entry_min <= z_v <= z_entry_max:
            side = "short"
        elif e9_v < e21_v and -z_entry_max <= z_v <= -z_entry_min:
            side = "long"

        if side is None:
            continue

        # ── 最小 TP 距離 ────────────────────────────────────────────────
        mean45 = closes[i - Z_WINDOW: i].mean()
        if side == "long":
            tp  = max(mean45, mark + atr_v * 0.5)
            sl  = mark - atr_v * sl_mult
        else:
            tp  = min(mean45, mark - atr_v * 0.5)
            sl  = mark + atr_v * sl_mult

        tp_dist = abs(tp - mark)
        sl_dist = abs(sl - mark)
        min_tp  = mark * ROUND_TRIP * 5.0
        if tp_dist < min_tp or sl_dist <= 0:
            continue
        if tp_dist / sl_dist < 0.5:   # 反趨勢 RR 放寬到 0.5
            continue

        pos_side    = side
        pos_entry   = mark
        pos_tp      = tp
        pos_sl      = sl
        pos_entry_i = i

    # ── 統計 ────────────────────────────────────────────────────────────
    if not pnls:
        return SimResult(0, 0, 0, 0, 0, 0, 0, 0, 0)

    p = np.array(pnls)
    wins = (p > 0).sum()
    eq = np.concatenate([[EQUITY], EQUITY + np.cumsum(p)])
    peak = np.maximum.accumulate(eq)
    mdd = float(((peak - eq) / np.where(peak > 0, peak, 1)).max()) * 100
    ann = math.sqrt(24 * 365)
    sharpe = float((p.mean() / (p.std() + 1e-12)) * ann)
    tp_r = sum(1 for r in reasons if r == "TP") / len(reasons) * 100
    sl_r = sum(1 for r in reasons if r == "SL") / len(reasons) * 100

    return SimResult(
        n=len(p),
        win_rate=round(wins / len(p) * 100, 1),
        total_pnl=round(p.sum(), 4),
        avg_pnl=round(p.mean(), 4),
        sharpe=round(sharpe, 3),
        mdd=round(mdd, 2),
        tp_rate=round(tp_r, 1),
        sl_rate=round(sl_r, 1),
        avg_hold=round(np.mean(hold_bars_list), 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Walk-Forward Engine
# ═══════════════════════════════════════════════════════════════════════════

def generate_param_combos() -> List[Dict]:
    keys = list(PARAM_GRID.keys())
    combos = []
    for vals in itertools.product(*[PARAM_GRID[k] for k in keys]):
        p = dict(zip(keys, vals))
        # 過濾無效組合
        if p["z_entry_min"] >= p["z_entry_max"]:
            continue
        combos.append(p)
    return combos


def run_wfa(
    df: pd.DataFrame,
    symbol: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    對單一 symbol 跑 WFA。
    返回 (all_oos_rows, best_per_fold_rows)
    """
    n = len(df)
    combos = generate_param_combos()
    total_bars = IS_BARS + OOS_BARS

    if n < total_bars + 100:
        logger.warning("  %s 數據不足 (%d bars)，跳過", symbol, n)
        return [], []

    # 計算 fold 起點（滾動 OOS_BARS）
    fold_starts = list(range(0, n - total_bars, OOS_BARS))
    logger.info("  %s: %d bars, %d folds, %d param combos",
                symbol, n, len(fold_starts), len(combos))

    all_oos: List[Dict] = []
    best_per_fold: List[Dict] = []

    for fold_i, start in enumerate(fold_starts):
        is_df  = df.iloc[start: start + IS_BARS].reset_index(drop=True)
        oos_df = df.iloc[start + IS_BARS: start + total_bars].reset_index(drop=True)

        if len(is_df) < IS_BARS // 2 or len(oos_df) < OOS_BARS // 4:
            continue

        # ── IS: 網格搜索 ────────────────────────────────────────────────
        best_is_sharpe = -999.0
        best_params = combos[0]

        for p in combos:
            r = simulate_counter_trend(is_df, **p)
            if r.n < MIN_TRADES_IS:
                continue
            if r.sharpe > best_is_sharpe:
                best_is_sharpe = r.sharpe
                best_params = p

        # ── OOS: 用最佳 IS 參數驗證 ────────────────────────────────────
        oos_r = simulate_counter_trend(oos_df, **best_params)

        row_base = {
            "symbol":       symbol,
            "fold":         fold_i,
            "is_start":     start,
            "is_end":       start + IS_BARS,
            "oos_start":    start + IS_BARS,
            "oos_end":      start + total_bars,
            "is_sharpe":    round(best_is_sharpe, 3),
            **{f"param_{k}": v for k, v in best_params.items()},
            "oos_n":        oos_r.n,
            "oos_sharpe":   oos_r.sharpe,
            "oos_total_pnl": oos_r.total_pnl,
            "oos_win_rate": oos_r.win_rate,
            "oos_mdd":      oos_r.mdd,
            "oos_tp_rate":  oos_r.tp_rate,
            "oos_sl_rate":  oos_r.sl_rate,
            "oos_avg_hold": oos_r.avg_hold,
        }
        best_per_fold.append(row_base)

        # 同時記錄所有 OOS 結果（供分析分佈用）
        for p in combos:
            r = simulate_counter_trend(oos_df, **p)
            all_oos.append({
                "symbol": symbol, "fold": fold_i,
                **{f"param_{k}": v for k, v in p.items()},
                "oos_n": r.n, "oos_sharpe": r.sharpe,
                "oos_total_pnl": r.total_pnl, "oos_mdd": r.mdd,
            })

        logger.info(
            "    fold %d  IS_sharpe=%.2f  best=%s  OOS: n=%d sharpe=%.2f pnl=%+.2f mdd=%.1f%%",
            fold_i, best_is_sharpe,
            {k: v for k, v in best_params.items()},
            oos_r.n, oos_r.sharpe, oos_r.total_pnl, oos_r.mdd,
        )

    return all_oos, best_per_fold


# ═══════════════════════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out",      default="wfa_result.csv")
    ap.add_argument("--best",     default="wfa_best.csv")
    ap.add_argument("--symbols",  default="",
                    help="逗號分隔，預設用 MR_SYMBOLS")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] or MR_SYMBOLS

    all_oos_rows: List[Dict] = []
    best_rows: List[Dict] = []

    for sym in syms:
        fname = sym.replace("/", "_").replace(":", "_") + ".parquet"
        fpath = data_dir / fname
        if not fpath.exists():
            logger.warning("找不到 %s，跳過", fpath)
            continue

        logger.info("[%s]", sym)
        df = pd.read_parquet(fpath)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = precompute(df)

        oos_rows, b_rows = run_wfa(df, sym)
        all_oos_rows.extend(oos_rows)
        best_rows.extend(b_rows)

    if not best_rows:
        logger.error("沒有有效結果，請確認 data/ 目錄有足夠歷史數據")
        return

    pd.DataFrame(all_oos_rows).to_csv(args.out, index=False)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(args.best, index=False)
    logger.info("OOS 明細 → %s", args.out)
    logger.info("最佳參數 → %s", args.best)

    # ── Aggregate 報告 ────────────────────────────────────────────────────
    W = 70
    print("\n" + "═" * W)
    print("  WALK-FORWARD RESULTS — 反趨勢 MR (ADX rising 反向入場)")
    print("═" * W)

    # 按 symbol 聚合 OOS
    for sym in syms:
        rows = best_df[best_df["symbol"] == sym]
        if rows.empty:
            continue
        valid = rows[rows["oos_n"] > 0]
        if valid.empty:
            print(f"  {sym:<30}  no OOS trades")
            continue
        avg_sh  = valid["oos_sharpe"].mean()
        avg_pnl = valid["oos_total_pnl"].mean()
        avg_mdd = valid["oos_mdd"].mean()
        n_folds = len(valid)
        pos_folds = (valid["oos_sharpe"] > 0).sum()
        print(f"  {sym:<30}  folds={n_folds}  "
              f"pos_folds={pos_folds}/{n_folds}  "
              f"avg_sharpe={avg_sh:+.2f}  "
              f"avg_pnl={avg_pnl:+.2f}U  "
              f"avg_mdd={avg_mdd:.1f}%")

    print("═" * W)

    # 跨 symbol 最穩定參數（出現最多次的最佳組合）
    param_cols = [c for c in best_df.columns if c.startswith("param_")]
    if param_cols:
        print("\n  最常出現的最佳參數組合（IS 選出，OOS 驗證）：")
        combo_counts = best_df[best_df["oos_sharpe"] > 0].groupby(param_cols).size()
        if not combo_counts.empty:
            top = combo_counts.sort_values(ascending=False).head(3)
            for combo, cnt in top.items():
                if isinstance(combo, tuple):
                    pdict = dict(zip([c.replace("param_","") for c in param_cols], combo))
                else:
                    pdict = {param_cols[0].replace("param_","") : combo}
                print(f"    count={cnt:>3}  {pdict}")

    print("═" * W)
    print()
    print("  下一步：把出現最多次的參數填入 atm_engine.py")
    print("  重點看 pos_folds / total_folds 比率，> 60% 才值得用")


if __name__ == "__main__":
    main()
