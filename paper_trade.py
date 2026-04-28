#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_trade.py — 連接真實 Hyperliquid 拉取即時數據，
                  但所有落單都在本地模擬，不會動用真實資金。

啟動方式：
    python paper_trade.py

環境變數（同 main_atm_1h.py，金鑰只用於讀取行情）：
    HYPERLIQUID_WALLET          錢包地址（必填）
    HYPERLIQUID_PRIVATE_KEY     私鑰（必填，HL 公開行情 REST 仍需認證）
    ATM_EQUITY_USDT             初始模擬資金，預設 600
    ATM_SYMBOLS                 交易對，逗號分隔
    ATM_POLL_SEC                輪詢間隔（秒），預設 2.0
    PT_LOG_FILE                 CSV 輸出路徑，預設 paper_trades.csv
    PT_DIAG_EVERY               診斷輸出間隔（秒），預設 30；設 0 關閉

ROUND5 changes vs ROUND4:
  - DIAG 行新增 exh_bars（連續低於 adx_chop 的 bar 數）和 vp_sell/vp_buy
    （volume proxy exhaustion 狀態），幫助調參時看清楚哪一關在擋
  - refresh_lee_ready() 不再在 DIAG 路徑中呼叫（REST stale-mid 問題已知）
  - LR_sell / LR_buy 欄位改為顯示 vp_sell / vp_buy（volume proxy）

成交模擬邏輯（Maker 訂單）：
    每次信號觸發時，以當時 orderbook 3層加權均價為成交價。
    假設：
        - maker rebate = +0.002%（HL VIP0）
        - 不考慮部分成交（整張成交或不成交）
        - 每個 symbol 同時只允許一個倉位
        - TP = Z-score 回到 0 附近（mark 回到 45m 均價）
        - SL = entry ± 1× ATR(14) on 1h

輸出：
    - 即時 console 顯示每筆模擬交易
    - paper_trades.csv：每筆平倉後記錄
    - 每 60 秒打印一次帳戶摘要
"""
from __future__ import annotations

import sys
import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import ccxt  # type: ignore
from dotenv import load_dotenv
load_dotenv()

from core.atm_engine import (
    ATMBot,
    ATMSessionConfig,
    MarketRegime,
    volume_proxy_exhaustion,  # *** ROUND5: imported for DIAG use
)
from core.indicators import atr, rolling_z_score, adx_series

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paper_trade")

# ── 模擬帳戶 ────────────────────────────────────────────────────────────────

@dataclass
class SimPosition:
    """單筆模擬倉位（一個 symbol 同時只能有一個）。"""
    symbol: str
    side: str           # "long" / "short"
    entry_price: float
    amount: float       # base qty
    entry_time: float   # wall time (time.time())
    tp_price: float     # 目標價
    sl_price: float     # 止損價
    regime: str
    adx_at_entry: float
    z_at_entry: float
    imb_at_entry: float
    mei: Optional[float] = None   # 記錄用，不影響邏輯
    pos_mult: float = 1.0         # 記錄用

    def unrealized_pnl(self, mark: float) -> float:
        if self.side == "long":
            return (mark - self.entry_price) * self.amount
        return (self.entry_price - mark) * self.amount

    def pnl_bps(self, mark: float) -> float:
        base = (mark - self.entry_price) / self.entry_price * 1e4
        return base if self.side == "long" else -base


@dataclass
class SimAccount:
    """模擬帳戶：資金、倉位、交易記錄。"""
    initial_equity: float
    maker_rebate: float = 0.00002   # +0.002% per fill
    taker_fee: float = 0.00035      # 不用，但記錄用

    equity: float = field(init=False)
    positions: Dict[str, SimPosition] = field(default_factory=dict, init=False)
    closed_trades: List[Dict] = field(default_factory=list, init=False)
    _trade_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.equity = self.initial_equity

    def open(self, pos: SimPosition) -> None:
        if pos.symbol in self.positions:
            logger.warning("Already have position in %s — skip new signal", pos.symbol)
            return
        fee = pos.entry_price * pos.amount * self.maker_rebate
        self.equity += fee
        self.positions[pos.symbol] = pos
        self._trade_counter += 1
        logger.info(
            "OPEN  #%d  %s %s  qty=%.6f  entry=%.4f  TP=%.4f  SL=%.4f  "
            "regime=%s  ADX=%.1f  Z=%.2f  imb=%.2f  "
            "MEI=%s  pos_mult=%.1f",
            self._trade_counter, pos.side.upper(), pos.symbol,
            pos.amount, pos.entry_price, pos.tp_price, pos.sl_price,
            pos.regime, pos.adx_at_entry, pos.z_at_entry, pos.imb_at_entry,
            f"{pos.mei:.3f}" if pos.mei is not None else "n/a", pos.pos_mult,
        )

    def close(self, symbol: str, exit_price: float, reason: str) -> Optional[Dict]:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        fee = exit_price * pos.amount * self.maker_rebate
        if pos.side == "long":
            gross_pnl = (exit_price - pos.entry_price) * pos.amount
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.amount
        net_pnl = gross_pnl + fee
        self.equity += net_pnl
        hold_min = (time.time() - pos.entry_time) / 60.0
        rec = {
            "trade_id": self._trade_counter,
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "amount": pos.amount,
            "gross_pnl": round(gross_pnl, 6),
            "fee_rebate": round(fee, 6),
            "net_pnl": round(net_pnl, 6),
            "hold_min": round(hold_min, 1),
            "reason": reason,
            "regime": pos.regime,
            "adx_entry": round(pos.adx_at_entry, 2),
            "z_entry": round(pos.z_at_entry, 3),
            "imb_entry": round(pos.imb_at_entry, 3),
            "equity_after": round(self.equity, 4),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.closed_trades.append(rec)
        logger.info(
            "CLOSE #%d  %s %s  exit=%.4f  net_pnl=%+.4f USDT  reason=%s  "
            "equity=%.2f",
            self._trade_counter, pos.side.upper(), symbol,
            exit_price, net_pnl, reason, self.equity,
        )
        return rec

    def summary(self) -> str:
        n = len(self.closed_trades)
        if n == 0:
            return f"equity={self.equity:.2f}  trades=0  open={len(self.positions)}"
        wins = sum(1 for t in self.closed_trades if t["net_pnl"] > 0)
        total_pnl = sum(t["net_pnl"] for t in self.closed_trades)
        avg = total_pnl / n
        return (
            f"equity={self.equity:.2f}  closed={n}  "
            f"win_rate={wins/n*100:.0f}%  total_pnl={total_pnl:+.4f}  "
            f"avg_pnl={avg:+.4f}  open={list(self.positions.keys())}"
        )


# ── 成交價計算（模擬 maker 3 層加權均價）──────────────────────────────────

def _sim_fill_price(ob: Dict, side: str, fallback: float) -> float:
    """
    模擬 maker 掛單成交價：
    買入 → bid 三層加權均價（我們掛在 bid 側，假設成交）
    賣出 → ask 三層加權均價
    """
    key = "bids" if side == "buy" else "asks"
    levels = (ob.get(key) or [])[:3]
    if not levels:
        return fallback
    total_sz = sum(lv[1] for lv in levels if len(lv) > 1 and lv[1])
    if total_sz <= 0:
        return float(levels[0][0]) if levels[0][0] else fallback
    return sum(lv[0] * lv[1] for lv in levels if len(lv) > 1 and lv[0] and lv[1]) / total_sz


def _compute_tp_sl(
    o1h: List[List[float]],
    o1m: List[List[float]],
    side: str,
    entry: float,
    regime: str = "chop",
) -> tuple[float, float]:
    """
    MR 路徑（chop / neutral）：
        TP = 45m 均價（Z 回歸目標）
        SL = entry ± 1.0 × ATR(14) on 1H

    TREND 路徑（已移除，保留說明）：
        舊版 TREND 跟隨已淘汰，改為反趨勢 MR（CTR_MR）。

    CTR_MR 路徑（ROUND8）：
        TP = 45m 均價回歸（同普通 MR，反趨勢預期價格回到均值）
        SL = entry ± ctr_sl_mult × ATR（WFA 最優 = 1.0×ATR）
    """
    a = atr(o1h, 14) or (entry * 0.005)
    is_ctr = "ctr_mr" in regime.lower()

    if is_ctr:
        # 反趨勢 MR：TP 回歸 45m 均價，SL = 1.0×ATR（WFA 優化值）
        closes_1m = [x[4] for x in o1m]
        window = min(45, len(closes_1m))
        tp_price = sum(closes_1m[-window:]) / window
        if side == "long":
            sl_price = entry - 1.0 * a
            tp_price = max(tp_price, entry + a * 0.5)
        else:
            sl_price = entry + 1.0 * a
            tp_price = min(tp_price, entry - a * 0.5)
    else:
        # 普通 MR：TP 回歸 45m 均價，SL 1×ATR
        closes_1m = [x[4] for x in o1m]
        window = min(45, len(closes_1m))
        tp_price = sum(closes_1m[-window:]) / window
        if side == "long":
            sl_price = entry - a
            tp_price = max(tp_price, entry + a * 0.5)
        else:
            sl_price = entry + a
            tp_price = min(tp_price, entry - a * 0.5)

    return tp_price, sl_price


# ── CSV 輸出 ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "trade_id", "symbol", "side", "entry_price", "exit_price", "amount",
    "gross_pnl", "fee_rebate", "net_pnl", "hold_min", "reason",
    "regime", "adx_entry", "z_entry", "imb_entry", "equity_after", "closed_at",
]


def _append_csv(path: str, rec: Dict) -> None:
    write_header = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in CSV_FIELDS})


# ── 主要 Paper Trade Loop ───────────────────────────────────────────────────

def _exchange_from_env() -> ccxt.Exchange:
    wallet = os.environ.get("HYPERLIQUID_WALLET", "").strip()
    private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "").strip()
    if not wallet or not private_key:
        raise RuntimeError(
            "請設定環境變數 HYPERLIQUID_WALLET 和 HYPERLIQUID_PRIVATE_KEY\n"
            "（用於讀取行情，不會下任何真實訂單）"
        )
    return ccxt.hyperliquid(
        {
            "walletAddress": wallet,
            "privateKey": private_key,
            "enableRateLimit": True,
        }
    )


class PaperTradeBot:
    """
    ATMBot 的 paper trade 包裝器。

    做法：
    1. 用 ATMBot.on_signal callback 攔截所有信號
    2. 在本地 SimAccount 開倉（不呼叫任何 create_order）
    3. 每次 tick 對所有未平倉倉位做 TP/SL/早退檢查
    4. 平倉後寫入 CSV
    """

    def __init__(
        self,
        ex: ccxt.Exchange,
        session: ATMSessionConfig,
        account: SimAccount,
        csv_path: str,
    ) -> None:
        self.account = account
        self.csv_path = csv_path
        self.bot = ATMBot(ex=ex, session=session, on_signal=self._on_signal)
        # ATM_ADX_CHOP: override RegimeFilter.adx_chop at runtime without touching engine code
        _adx_chop_env = os.environ.get("ATM_ADX_CHOP", "")
        if _adx_chop_env:
            self.bot.regime.adx_chop = float(_adx_chop_env)
            logger.info("RegimeFilter.adx_chop overridden to %.1f via ATM_ADX_CHOP", self.bot.regime.adx_chop)
        self._last_summary_t = time.time()
        self._ex = ex
        self._session = session
        self._ohlcv_cache: Dict[str, Dict[str, List]] = {}

    # ── Signal callback（攔截 ATMBot 的落單意圖）──────────────────────────

    def _on_signal(self, sig: Dict[str, Any]) -> None:
        symbol = sig["symbol"]
        side = sig["side"]
        amount = sig["amount"]
        mark = sig["mark"]
        regime = str(sig.get("entry_mode", "") or sig.get("regime", "chop")).lower()

        if symbol in self.account.positions:
            return

        try:
            ob = self._ex.fetch_order_book(symbol, 3)
        except Exception as e:
            logger.warning("OB fetch failed for %s: %s", symbol, e)
            ob = {}
        cs = "buy" if side == "long" else "sell"
        fill_price = _sim_fill_price(ob, cs, mark)

        cache = self._ohlcv_cache.get(symbol, {})
        o1h = cache.get("1h", [])
        o1m = cache.get("1m", [])
        if not o1h or not o1m:
            logger.warning("No OHLCV cache for %s — skip signal", symbol)
            return

        # *** ROUND5.1: pass regime so TREND uses ATR-based TP/SL
        tp_price, sl_price = _compute_tp_sl(o1h, o1m, side, fill_price, regime=regime)

        mei      = sig.get("mei")
        pos_mult = sig.get("pos_mult", 1.0)
        # ── 最小利潤門檻 ────────────────────────────────────────────────────
        # TP 距離必須 > min_tp_fee_multiple × round-trip fee，
        # 否則即使全部 TP 也只是在養交易所，沒有實質 edge。
        tp_dist = abs(tp_price - fill_price)
        sl_dist = abs(sl_price - fill_price)
        min_tp_dist = fill_price * self._session.round_trip_fee_rate * self._session.min_tp_fee_multiple
        if tp_dist < min_tp_dist:
            logger.info(
                "SKIP  %s %s  tp_dist=%.6f < min=%.6f (%.1f× fee)  regime=%s",
                side.upper(), symbol, tp_dist, min_tp_dist,
                self._session.min_tp_fee_multiple, regime,
            )
            return
        # R:R 檢查（沿用現有 rr_vs_fees_ok 邏輯）
        if sl_dist > 0 and (tp_dist / sl_dist) < self._session.rr_min:
            logger.info(
                "SKIP  %s %s  rr=%.2f < min=%.2f  regime=%s",
                side.upper(), symbol, tp_dist / sl_dist, self._session.rr_min, regime,
            )
            return

        z_val = sig.get("z") or 0.0
        closes_1m = [x[4] for x in o1m]
        zs = rolling_z_score(closes_1m, self._session.z_window)
        if zs:
            z_val = zs[0]

        pos = SimPosition(
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            amount=amount,
            entry_time=time.time(),
            tp_price=tp_price,
            sl_price=sl_price,
            regime=regime,
            adx_at_entry=float(sig.get("adx") or 0.0),
            z_at_entry=float(z_val),
            imb_at_entry=float(sig.get("imb") or 0.0),
            mei=sig.get("mei"),
            pos_mult=float(sig.get("pos_mult", 1.0)),
        )
        self.account.open(pos)

    # ── 純 Z-score 入場 loop（無 regime / MEI / volume-proxy / imbalance gate）──

    def _simple_z_entry_loop(self, symbols: List[str]) -> None:
        """
        最精簡 MR 入場：
          Entry : |Z| ∈ [z_entry_min, z_entry_max]（1m rolling Z-score）
          TP    : 45m mean（Z 回歸 0）
          SL    : entry ± 1× ATR(14, 1H)
        所有其他 Gate（regime、exhaustion、volume proxy、MEI、imbalance）全部移除。
        """
        z_min = self._session.z_entry_min
        z_max = self._session.z_entry_max
        z_win = self._session.z_window

        for symbol in symbols:
            if symbol in self.account.positions:
                continue

            cache = self._ohlcv_cache.get(symbol, {})
            o1m = cache.get("1m", [])
            o1h = cache.get("1h", [])
            if not o1m or not o1h:
                continue

            closes_1m = [x[4] for x in o1m]
            zs = rolling_z_score(closes_1m, z_win)
            if not zs:
                continue
            z_val, _z_mean, _z_std = zs

            side: Optional[str] = None
            if -z_max <= z_val <= -z_min:
                side = "long"
            elif z_min <= z_val <= z_max:
                side = "short"

            if side is None:
                continue

            # 使用緩存的最新 1m close 作為成交價（紙交易不需額外 REST）
            fill_price = float(closes_1m[-1])

            tp_price, sl_price = _compute_tp_sl(o1h, o1m, side, fill_price)

            notional = self._session.equity_usdt * self._session.risk_fraction_per_symbol * 8.0
            raw_amt = notional / max(fill_price, 1e-9)
            try:
                amount = float(self._ex.amount_to_precision(symbol, raw_amt))
            except Exception:
                amount = round(raw_amt, 6)
            if amount <= 0:
                continue

            pos = SimPosition(
                symbol=symbol,
                side=side,
                entry_price=fill_price,
                amount=amount,
                entry_time=time.time(),
                tp_price=tp_price,
                sl_price=sl_price,
                regime="z_only",
                adx_at_entry=0.0,
                z_at_entry=float(z_val),
                imb_at_entry=0.0,
            )
            self.account.open(pos)

    # ── 倉位管理（每 tick 檢查）────────────────────────────────────────────

    def _check_positions(self) -> None:
        for symbol in list(self.account.positions.keys()):
            pos = self.account.positions[symbol]
            # 使用緩存的最新 1m close，避免每 tick 打 fetch_ticker
            o1m_cache = self._ohlcv_cache.get(symbol, {}).get("1m", [])
            if o1m_cache:
                mark = float(o1m_cache[-1][4])
            else:
                mark = pos.entry_price

            reason = None

            if pos.side == "long" and mark >= pos.tp_price:
                reason = "TP"
            elif pos.side == "short" and mark <= pos.tp_price:
                reason = "TP"

            if pos.side == "long" and mark <= pos.sl_price:
                reason = "SL"
            elif pos.side == "short" and mark >= pos.sl_price:
                reason = "SL"

            if reason is None and self.bot.should_early_exit(symbol, mark, pos.entry_price, pos.side):
                reason = "DECEL"

            if reason:
                rec = self.account.close(symbol, mark, reason)
                if rec:
                    _append_csv(self.csv_path, rec)

    # ── 每 tick 診斷（顯示每個 symbol 卡在哪一關）──────────────────────────

    def _log_diagnostics(self, symbols: List[str], now: datetime) -> None:
        """
        *** ROUND5: DIAG 行新增兩個欄位：
          exh_bars  — 連續低於 adx_chop 的 ADX bar 數（Gate 4 計數器可視化）
          vp_sell   — volume_proxy_exhaustion(o1m, 'long')  的結果（取代 LR_sell）
          vp_buy    — volume_proxy_exhaustion(o1m, 'short') 的結果（取代 LR_buy）

        也移除了 refresh_lee_ready() 呼叫（REST stale-mid 問題，見 atm_engine.py [B]）。
        設 PT_DIAG_EVERY=0 可關閉診斷輸出。
        """
        in_window = self.bot.in_soft_trade_window(now, self._session)
        if not in_window:
            logger.info("DIAG  時間窗口：關閉（minute%%30=%d，週末cap=48）", now.minute % 30)
            return

        for symbol in symbols:
            try:
                if self.bot.vol_guard.is_frozen(symbol):
                    remain = self.bot.vol_guard.remaining(symbol)
                    logger.info("DIAG  %-20s  VolGuard 凍結中 還剩 %.0fs", symbol, remain)
                    continue

                cache = self._ohlcv_cache.get(symbol, {})
                o1h = cache.get("1h", [])
                o1m = cache.get("1m", [])
                if not o1h or not o1m:
                    logger.info("DIAG  %-20s  OHLCV 未就緒", symbol)
                    continue

                # Regime
                regime, adx_val, rv = self.bot.regime.evaluate(o1h)
                exhaustion = self.bot.regime.exhaustion_ok(o1h) if regime in (
                    MarketRegime.CHOP, MarketRegime.NEUTRAL) else None

                # *** ROUND5: exh_bars — count consecutive bars below adx_chop
                raw_adx = adx_series(o1h, self.bot.regime.adx_period)
                exh_bars = 0
                for v in reversed(raw_adx):
                    if v < self.bot.regime.adx_chop:
                        exh_bars += 1
                    else:
                        break

                # *** ROUND5: adx_mom diagnostics (rate/accel) — not used in entry, shown for tuning
                mom = self.bot.regime.exhaustion_diagnostics(o1h)
                mom_str = (
                    f"rate={mom[0]:+.3f} accel={mom[1]:+.3f}"
                    if mom else "mom=n/a"
                )

                # Z-score
                closes_1m = [x[4] for x in o1m]
                zs = rolling_z_score(closes_1m, self._session.z_window)
                z_val, z_mean, z_std = (zs[0], zs[1], zs[2]) if zs else (None, None, None)

                z_long_entry  = (z_mean - self._session.z_entry_min  * z_std) if zs else None
                z_short_entry = (z_mean + self._session.z_entry_min  * z_std) if zs else None

                ticker = self._ex.fetch_ticker(symbol)
                mark = float(ticker.get("last") or o1h[-1][4])

                imb = self.bot.orderbook_imbalance(symbol)
                funding = self.bot.predicted_funding_rate(symbol)

                from core.indicators import atr as _atr
                atr_val = _atr(o1h, 14)
                atr_bps = (atr_val / mark * 1e4) if atr_val and mark else None

                # *** ROUND5: volume proxy exhaustion (replaces LR_sell / LR_buy)
                vp_sell = volume_proxy_exhaustion(
                    o1m, "long",
                    window=self._session.vol_proxy_window,
                    decay_ratio=self._session.vol_proxy_decay,
                )
                vp_buy = volume_proxy_exhaustion(
                    o1m, "short",
                    window=self._session.vol_proxy_window,
                    decay_ratio=self._session.vol_proxy_decay,
                )

                gap_long  = (mark - z_long_entry)  if z_long_entry  is not None else None
                gap_short = (z_short_entry - mark) if z_short_entry is not None else None

                # MEI for DIAG
                try:
                    diag_mei = self.bot._compute_mei(o1h)
                    mei_str = f"{diag_mei:.3f}" if diag_mei is not None else "n/a"
                except Exception:
                    mei_str = "n/a"

                logger.info(
                    "DIAG  %-20s  mark=%-9.2f  Z=%-6s  std=%-6s  "
                    "long觸發<%-9s  short觸發>%-9s  "
                    "距long=%-7s  距short=%-7s  "
                    "regime=%-8s  ADX=%-5.1f  exh=%s(bars=%d)  %s  MEI=%s  "
                    "imb=%+.2f  ATR=%-5s bps  funding=%.4f%%  vp_sell=%s  vp_buy=%s",
                    symbol, mark,
                    f"{z_val:.3f}"   if z_val  is not None else "None",
                    f"{z_std:.2f}"   if z_std  is not None else "None",
                    f"{z_long_entry:.2f}"  if z_long_entry  is not None else "None",
                    f"{z_short_entry:.2f}" if z_short_entry is not None else "None",
                    f"{gap_long:+.2f}"  if gap_long  is not None else "None",
                    f"{gap_short:+.2f}" if gap_short is not None else "None",
                    str(regime.value) if regime else "None",
                    adx_val or 0,
                    str(exhaustion) if exhaustion is not None else "N/A",
                    exh_bars,
                    mom_str,
                    mei_str,
                    imb,
                    f"{atr_bps:.1f}" if atr_bps else "None",
                    (funding or 0) * 100,
                    vp_sell, vp_buy,
                )

            except Exception as e:
                logger.warning("DIAG  %s  error: %s", symbol, e)

    # ── 主迴路 ─────────────────────────────────────────────────────────────

    def run(self, symbols: List[str], poll_sec: float) -> None:
        logger.info(
            "Paper Trade 啟動  equity=%.2f USDT  symbols=%s  poll=%.1fs",
            self.account.equity, symbols, poll_sec,
        )
        logger.info("CSV 輸出 → %s", os.path.abspath(self.csv_path))
        logger.info("（不會產生任何真實訂單）")

        diag_every = float(os.environ.get("PT_DIAG_EVERY", "30"))
        last_diag_t = 0.0
        fetch_workers = min(int(os.environ.get("PT_FETCH_WORKERS", "5")), len(symbols))

        def _fetch_one(sym: str) -> None:
            """帶 429 退避重試的 OHLCV 拉取（最多 3 次，1s / 2s 間隔）。"""
            for attempt in range(3):
                try:
                    o1m = [list(x) for x in self._ex.fetch_ohlcv(sym, "1m", limit=200)]
                    time.sleep(0.15)   # 同 symbol 兩個 timeframe 之間留間隙
                    o1h = [list(x) for x in self._ex.fetch_ohlcv(sym, "1h", limit=200)]
                    self._ohlcv_cache[sym] = {"1m": o1m, "1h": o1h}
                    return
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        wait = 2 ** attempt   # 1s → 2s
                        logger.warning("429 on %s, retry in %ds (attempt %d)", sym, wait, attempt + 1)
                        time.sleep(wait)
                    else:
                        logger.warning("OHLCV fetch error %s: %s", sym, e)
                        return

        while True:
            now = datetime.now(timezone.utc)

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
                futs = {pool.submit(_fetch_one, sym): sym for sym in symbols}
                for fut in as_completed(futs):
                    pass

            try:
                self._simple_z_entry_loop(symbols)
            except Exception as e:
                logger.exception("z_entry_loop error: %s", e)

            if diag_every > 0 and (time.time() - last_diag_t) >= diag_every:
                self._log_diagnostics(symbols, now)
                last_diag_t = time.time()

            self._check_positions()

            if time.time() - self._last_summary_t >= 60.0:
                logger.info("── 帳戶摘要 %s", self.account.summary())
                self._last_summary_t = time.time()

            time.sleep(poll_sec)


# ── 入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    session = ATMSessionConfig(
        equity_usdt=float(os.environ.get("ATM_EQUITY_USDT", "600")),
        taker_fee_rate=float(os.environ.get("TAKER_FEE", "0.00035")),
        round_trip_fee_rate=float(os.environ.get("ROUND_TRIP_FEE", "0.0003")),
        z_window=int(os.environ.get("ATM_Z_WINDOW", "45")),
        z_entry_min=float(os.environ.get("ATM_Z_ENTRY_MIN", "1.5")),
        z_entry_max=float(os.environ.get("ATM_Z_ENTRY_MAX", "2.5")),
        min_abs_imbalance=float(os.environ.get("ATM_IMB_MIN", "0.03")),
        rr_min=float(os.environ.get("ATM_RR_MIN", "1.0")),
        min_tp_fee_multiple=float(os.environ.get("ATM_MIN_TP_FEE_MULT", "3.0")),
        maker_rebate_rate=float(os.environ.get("MAKER_REBATE", "-0.000384")),
        vol_proxy_window=int(os.environ.get("ATM_VP_WINDOW", "10")),
        vol_proxy_decay=float(os.environ.get("ATM_VP_DECAY", "0.90")),
    )
    ex = _exchange_from_env()
    ex.load_markets()

    from core.atm_engine import DEFAULT_SYMBOLS
    symbols_raw = os.environ.get("ATM_SYMBOLS", DEFAULT_SYMBOLS)
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
    poll_sec = float(os.environ.get("ATM_POLL_SEC", "2.0"))
    csv_path = os.environ.get("PT_LOG_FILE", "paper_trades.csv")

    account = SimAccount(initial_equity=session.equity_usdt)
    bot = PaperTradeBot(ex=ex, session=session, account=account, csv_path=csv_path)
    bot.run(symbols=symbols, poll_sec=poll_sec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBYE !!!")
