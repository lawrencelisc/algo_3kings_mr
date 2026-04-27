#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_ohlcv.py — 從 Hyperliquid 拉取 Top-20 symbol 的 5m OHLCV，存成 parquet。

用法：
    python fetch_ohlcv.py

環境變數：
    HYPERLIQUID_WALLET        錢包地址
    HYPERLIQUID_PRIVATE_KEY   私鑰
    FETCH_SINCE_DAYS          回溯天數，預設 180（6 個月）
    FETCH_SYMBOLS             逗號分隔，預設自動拉 top-20 by open interest
    FETCH_TIMEFRAME           預設 5m
    FETCH_OUT_DIR             輸出目錄，預設 ./data

輸出：
    data/{symbol_safe}.parquet
    每個 parquet 欄位：timestamp(ms int64), open, high, low, close, volume

注意：
    HL REST 每次最多回傳 500 根 bar（5m × 500 = 41.7h）。
    腳本自動分批往回拉直到 since 為止，每批之間 sleep 0.25s 避免 rate limit。
"""
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt  # type: ignore
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_ohlcv")

# ── 設定 ──────────────────────────────────────────────────────────────────

TIMEFRAME   = os.environ.get("FETCH_TIMEFRAME", "1h")
SINCE_DAYS  = int(os.environ.get("FETCH_SINCE_DAYS", "180"))
OUT_DIR     = Path(os.environ.get("FETCH_OUT_DIR", "./data"))
BATCH       = 500          # HL 單次上限
SLEEP_S     = 0.25         # 批次間 sleep
TOP_N       = 20


def _exchange() -> ccxt.Exchange:
    wallet  = os.environ.get("HYPERLIQUID_WALLET", "").strip()
    privkey = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "").strip()
    if not wallet or not privkey:
        raise RuntimeError("請設定 HYPERLIQUID_WALLET 和 HYPERLIQUID_PRIVATE_KEY")
    return ccxt.hyperliquid({
        "walletAddress": wallet,
        "privateKey":    privkey,
        "enableRateLimit": True,
    })


def top20_by_oi(ex: ccxt.Exchange) -> list[str]:
    """
    用 fetch_tickers 拿 open interest，取前 20。
    HL 的 ticker info 包含 openInterestValue（USD）。
    """
    logger.info("拉取 ticker 列表以決定 Top-20 OI...")
    tickers = ex.fetch_tickers()
    oi_list = []
    for sym, t in tickers.items():
        if not sym.endswith(":USDC"):
            continue
        # ccxt 統一欄位是 quoteVolume 或 info.openInterestValue
        oi = None
        info = t.get("info") or {}
        oi_str = info.get("openInterestValue") or info.get("oi")
        if oi_str is not None:
            try:
                oi = float(oi_str)
            except (ValueError, TypeError):
                pass
        if oi is None:
            oi = float(t.get("quoteVolume") or 0)
        oi_list.append((sym, oi))

    oi_list.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in oi_list[:TOP_N]]
    logger.info("Top-%d symbols: %s", TOP_N, top)
    return top


# timeframe → 毫秒換算（用於 since 步進）
_TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000,
    "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "1d": 86_400_000,
}


def fetch_symbol(
    ex: ccxt.Exchange,
    symbol: str,
    since_ms: int,
    timeframe: str,
) -> pd.DataFrame:
    """
    用 since 參數正向拉取（舊→新），每批 BATCH 根。
    ccxt 的 since 在所有 exchange 都有效；endTime 是 HL 私有參數，
    ccxt 不一定正確轉發，導致舊版無限循環。

    換算邏輯：
      每批拉完後，since_ms += BATCH × bar_ms，繼續往後直到超過 now。
    """
    tf_ms = _TF_MS.get(timeframe)
    if tf_ms is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    all_bars: list[list] = []
    cursor = since_ms

    while True:
        try:
            bars = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=BATCH)
        except Exception as e:
            logger.warning("fetch_ohlcv error %s: %s — retry in 2s", symbol, e)
            time.sleep(2)
            continue

        if not bars:
            break

        all_bars.extend(bars)
        newest_ts = bars[-1][0]
        cursor = newest_ts + tf_ms   # 下一批從這根之後開始

        fetched_to = datetime.fromtimestamp(newest_ts / 1000, tz=timezone.utc)
        logger.info("  %s  fetched to %s  total_bars=%d",
                    symbol, fetched_to.strftime("%Y-%m-%d %H:%M"), len(all_bars))

        now_ms = int(time.time() * 1000)
        if newest_ts >= now_ms - tf_ms * 2:
            break   # 已到最新

        if len(bars) < BATCH:
            break   # 最後一批不足，說明沒有更多數據

        time.sleep(SLEEP_S)

    if not all_bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df[df["timestamp"] >= since_ms].copy()
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def symbol_to_filename(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_") + ".parquet"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ex = _exchange()
    ex.load_markets()

    env_syms = os.environ.get("FETCH_SYMBOLS", "").strip()
    if env_syms:
        symbols = [s.strip() for s in env_syms.split(",") if s.strip()]
    else:
        symbols = top20_by_oi(ex)

    since_dt  = datetime.now(tz=timezone.utc) - timedelta(days=SINCE_DAYS)
    since_ms  = int(since_dt.timestamp() * 1000)
    logger.info("拉取範圍：%s → now  timeframe=%s", since_dt.strftime("%Y-%m-%d"), TIMEFRAME)

    for i, sym in enumerate(symbols, 1):
        out_path = OUT_DIR / symbol_to_filename(sym)
        logger.info("[%d/%d] %s → %s", i, len(symbols), sym, out_path)

        df = fetch_symbol(ex, sym, since_ms, TIMEFRAME)
        if df.empty:
            logger.warning("  %s 沒有數據，跳過", sym)
            continue

        df.to_parquet(out_path, index=False)
        span_days = (df["timestamp"].max() - df["timestamp"].min()) / 86400_000
        logger.info("  saved %d bars  (%.1f days)  →  %s", len(df), span_days, out_path)

    logger.info("完成。數據目錄：%s", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
