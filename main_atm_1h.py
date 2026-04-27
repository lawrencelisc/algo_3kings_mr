#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1H Hyperliquid ATM — 程式進入點（預設 600 USDT 紙上交易；可改環境變數）。

金鑰：請用環境變數，勿寫入會提交的 YAML。
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import ccxt  # type: ignore

from core.atm_engine import ATMBot, ATMSessionConfig

# 日誌：INFO；內層迴路已避免每秒 DEBUG 洗版
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_atm_1h")


def _exchange_from_env() -> ccxt.Exchange:
    # Hyperliquid 錢包／金鑰（可依部署改用 Vault 等託管名稱）
    wallet = os.environ.get("HYPERLIQUID_WALLET", "").strip() or os.environ.get("PT_API_KEY", "")
    private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "").strip() or os.environ.get("PT_SECRET_KEY", "")
    if not wallet or not private_key:
        raise RuntimeError("Set HYPERLIQUID_WALLET and HYPERLIQUID_PRIVATE_KEY (or PT_API_KEY/PT_SECRET_KEY).")
    ex = ccxt.hyperliquid(
        {
            "walletAddress": wallet,
            "privateKey": private_key,
            "enableRateLimit": True,
        }
    )
    return ex


def main() -> None:
    session = ATMSessionConfig(
        equity_usdt=float(os.environ.get("ATM_EQUITY_USDT", "600")),
        taker_fee_rate=float(os.environ.get("TAKER_FEE", "0.00035")),
        round_trip_fee_rate=float(os.environ.get("ROUND_TRIP_FEE", "0.0003")),
    )
    ex = _exchange_from_env()
    ex.load_markets()
    # 交易標的（可縮成你願意承擔的範圍；HL 上主流對流動性較好）
    # ROUND6: 白名單 symbols（從 atm_engine.DEFAULT_SYMBOLS 讀取，env var 可覆蓋）
    from core.atm_engine import DEFAULT_SYMBOLS
    symbols = os.environ.get("ATM_SYMBOLS", DEFAULT_SYMBOLS).split(",")
    symbols = [s.strip() for s in symbols if s.strip()]
    bot = ATMBot(ex=ex, session=session, on_signal=None)
    poll_s = float(os.environ.get("ATM_POLL_SEC", "2.0"))
    logger.info("Starting ATM 1H loop symbols=%s poll_s=%.1f", symbols, poll_s)
    while True:
        try:
            t = datetime.now(timezone.utc)
            bot.atm_trigger_loop(symbols, now=t)
        except Exception as e:  # noqa: BLE001
            # 單次 REST 失敗不應結束主迴圈（實盤／紙上皆同）
            logger.exception("atm_trigger_loop error: %s", e)
        time.sleep(poll_s)


if __name__ == "__main__":
    main()
