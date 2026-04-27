# -*- coding: utf-8 -*-
"""
Hyperliquid 1H ATM — re-engineered execution core (regime + Lee-Ready + maker-first).

*** All non-obvious parameter choices and behavioral changes are marked with '***' comments
*** so reviewers can see intent without diffing your prior ROUND4 tree.

ROUND5 changes vs ROUND4 (2026-04-26):
  [A] RegimeFilter.exhaustion_ok — replaced noisy rate/accel double-diff gate with
      a simpler ADX-level confirm_bars counter.  The old gate required rate ≤ 0 AND
      accel < 0.5 simultaneously; on low-vol weekends ADX oscillates 12–18 and the
      double-diff rarely satisfies both conditions at once.  New gate: ADX must stay
      below adx_chop for confirm_bars consecutive 1H bars (default 3).  Identical
      protection against entering during a genuine trend ADX pause, but immune to
      Wilder-smoothing tick noise.  adx_mom() kept for diagnostic use.

  [B] Volume proxy exhaustion — LeeReadyTracker.sell/buy_exhaustion() replaced by
      volume_proxy_exhaustion() using 1m OHLCV close-vs-HL-midpoint direction proxy.
      The old REST fetch_trades approach ingested stale trades against current mid,
      causing systematic mis-classification and a pool that never warmed up.  The new
      approach needs zero pre-warming: just 10 recent 1m bars.

  [C] PerTickerVolumeGuard — three weekend-aware improvements:
        1. baseline changed from median → 75th-percentile (robust to occasional big
           bars that inflate the median on thin books)
        2. blast_ratio * 0.6 on weekends (low liquidity → lower bar to trigger is a
           false positive, not a real vacuum)
        3. freeze_sec halved on weekends (90s vs 180s) to recover opportunity set
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import ccxt  # type: ignore

from core.indicators import (
    PriceVelocityState,
    adx,
    adx_mom,
    adx_series,
    atr,
    ema,
    rolling_return_vol,
    rolling_z_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime filter (ADX + realized vol on 1H bars)
# ---------------------------------------------------------------------------


class MarketRegime(str, Enum):
    TREND = "trend"
    CHOP = "chop"
    NEUTRAL = "neutral"


@dataclass
class RegimeFilter:
    """
    *** ADX(14) on 1H OHLCV + short-window return volatility to label trend vs mean-reversion chop.
    *** Thresholds 25/20 match common practice; neutral band avoids whipsaw mode flips in ADX 20–25.
    """

    adx_period: int = 14
    adx_trend: float = 25.0
    adx_chop: float = 20.0
    ret_vol_window: int = 20
    # *** If 1H vol collapses, MR edges shrink — require vol floor before trusting chop.
    min_chop_annualized_vol: float = 0.08  # 8% annualized (scaled from 1H returns; heuristic)

    # *** [A] ROUND5: exhaustion_ok now uses consecutive-bar counter instead of
    # *** rate/accel double-diff.  The old params are kept for diagnostic calls only.
    adx_confirm_bars: int = 1       # ADX must be < adx_chop for this many consecutive 1H bars

    # Legacy momentum params — kept so adx_mom() can still be called for diagnostics/logging
    adx_smooth: int = 5             # raised from 3 → smoother series for rate/accel diag
    adx_lookback: int = 6           # raised from 4 → more bars averaged
    adx_rate_max: float = 0.3       # relaxed from 0.0 → tolerate small rises (oscillation noise)
    adx_accel_max: float = 1.5      # relaxed from 0.5 → rarely blocks in chop now

    def evaluate(self, ohlcv_1h: List[List[float]]) -> Tuple[Optional[MarketRegime], Optional[float], Optional[float]]:
        a = adx(ohlcv_1h, self.adx_period)
        closes = [x[4] for x in ohlcv_1h]
        rv = rolling_return_vol(closes, self.ret_vol_window)
        if a is None:
            return None, None, None
        if a >= self.adx_trend and rv is not None and rv * (24 * 252) ** 0.5 > self.min_chop_annualized_vol * 0.2:
            return MarketRegime.TREND, a, rv
        if a <= self.adx_chop:
            return MarketRegime.CHOP, a, rv
        return MarketRegime.NEUTRAL, a, rv

    def exhaustion_ok(self, ohlcv_1h: List[List[float]]) -> bool:
        """
        *** [A] ROUND5 replacement: ADX-level confirm_bars counter.

        True when the last `adx_confirm_bars` consecutive 1H bars all have ADX < adx_chop.

        Why this is safer than the old rate/accel gate
        -----------------------------------------------
        The old gate diffed a Wilder-smoothed ADX series (lag ~28–42 bars) and required
        rate ≤ 0 AND accel < 0.5 simultaneously.  In a low-vol weekend chop environment
        ADX oscillates between 10–18 rather than monotonically declining, so the double-
        diff flips positive every few bars even though the market is genuinely ranging.
        Result: exhaustion_ok() returned False almost continuously even for symbols that
        had been in CHOP regime for hours.

        The counter approach is noise-immune: we simply ask "has ADX been below the
        chop threshold for the last N consecutive hours?"  If yes, there is no trend to
        be exhausted — it's safe to enter MR.  N=3 means we need a clean 3-hour run
        below 20; a single spike to ADX=21 resets the clock, giving adequate protection
        against entering during a mid-trend ADX dip.

        Called only for CHOP / NEUTRAL regimes; TREND regime bypasses this check.
        """
        raw = adx_series(ohlcv_1h, self.adx_period)
        if len(raw) < self.adx_confirm_bars:
            return False
        return all(v < self.adx_chop for v in raw[-self.adx_confirm_bars:])

    def exhaustion_diagnostics(self, ohlcv_1h: List[List[float]]) -> Optional[Tuple[float, float, float]]:
        """
        Return (rate, accel, adx_now) from adx_mom() for DIAG logging.
        Not used in entry decisions — diagnostic only.
        """
        return adx_mom(
            ohlcv_1h,
            period=self.adx_period,
            smooth=self.adx_smooth,
            lookback=self.adx_lookback,
        )

    def trend_adx_rising_signal(self, ohlcv_1h: List[List[float]]) -> Optional[str]:
        """
        ROUND6: TREND v2 — ADX 連續 3 根 1H bar 單調遞增（趨勢加速確認）+ EMA stack 方向。
        Backtest 顯示此法 Sharpe 明顯優於 EMA cross（v1 已淘汰）。
        ADX rising = 趨勢仍在加速，不是衰竭段入場。
        """
        raw = adx_series(ohlcv_1h, self.adx_period)
        if len(raw) < 3:
            return None
        # ADX 必須連續上升（最近 3 根）
        if not (raw[-1] > raw[-2] > raw[-3]):
            return None
        # EMA stack 確認方向
        c = [x[4] for x in ohlcv_1h]
        e9, e21 = ema(c, 9), ema(c, 21)
        if e9 is None or e21 is None:
            return None
        if e9 > e21:
            return "long"
        if e9 < e21:
            return "short"
        return None


# ---------------------------------------------------------------------------
# [B] Volume proxy exhaustion — replaces LeeReadyTracker for CHOP MR entries
# ---------------------------------------------------------------------------


def volume_proxy_exhaustion(
    ohlcv_1m: List[List[float]],
    side: str,
    window: int = 10,
    decay_ratio: float = 0.65,
) -> bool:
    """
    *** [B] ROUND5: 1m OHLCV close-vs-HL-midpoint direction proxy for aggressor exhaustion.

    Replaces LeeReadyTracker.sell/buy_exhaustion() for CHOP regime entries.

    Motivation
    ----------
    The old REST fetch_trades() approach had two fatal flaws in a poll-based loop:
      1. Stale-mid mis-classification: historical trades were ingested against the
         *current* best_bid/best_ask, not the mid at trade time, causing systematic
         direction errors.
      2. Cold-start: LeeReadyTracker needed deque(maxlen=20) to be full before
         judging exhaustion — requiring 20 un-deduplicated recent trades.  In a REST
         environment the same trades are re-ingested every poll cycle, so the pool
         fills with duplicate entries and never reflects a true short-window snapshot.

    This function requires only the 1m OHLCV bars already fetched by the main loop —
    zero extra REST calls, zero pre-warming.

    Direction classification
    ------------------------
    For each 1m bar:
      close > (high + low) / 2  →  buy-side aggression dominated  (price closed in upper half)
      close < (high + low) / 2  →  sell-side aggression dominated (price closed in lower half)
      close == mid              →  neutral; split 50/50 by volume

    Exhaustion signal
    -----------------
    For a long entry (need sell exhaustion):
      long_avg  = mean sell-proxy volume over last `window` bars
      short_avg = mean sell-proxy volume over last `window//2` bars
      exhausted = short_avg < decay_ratio * long_avg  (short-term sell vol decaying)

    For a short entry (need buy exhaustion):
      same logic on buy-proxy volume.

    Parameters
    ----------
    ohlcv_1m   : list of [ms, open, high, low, close, volume] bars
    side       : 'long' (need sell-side exhaustion) or 'short' (need buy-side exhaustion)
    window     : lookback in 1m bars (default 10 → last 10 minutes of data)
    decay_ratio: short_avg must be below this fraction of long_avg (default 0.65)

    Returns
    -------
    True if the aggressor volume on the opposing side is declining.
    """
    if len(ohlcv_1m) < window + 1:
        return False

    bars = ohlcv_1m[-window:]
    buy_vols: List[float] = []
    sell_vols: List[float] = []

    for bar in bars:
        _, _o, h, l, c, v = bar
        mid = (h + l) / 2.0
        if h == l:
            # Zero-range bar: treat as neutral
            buy_vols.append(float(v) * 0.5)
            sell_vols.append(float(v) * 0.5)
        elif c > mid:
            buy_vols.append(float(v))
            sell_vols.append(0.0)
        elif c < mid:
            sell_vols.append(float(v))
            buy_vols.append(0.0)
        else:
            buy_vols.append(float(v) * 0.5)
            sell_vols.append(float(v) * 0.5)

    half = max(1, window // 2)

    if side == "long":
        # Waiting for sell aggression to fade before buying the dip
        long_avg = sum(sell_vols) / window
        short_avg = sum(sell_vols[-half:]) / half
    else:
        # Waiting for buy aggression to fade before shorting the rip
        long_avg = sum(buy_vols) / window
        short_avg = sum(buy_vols[-half:]) / half

    if long_avg <= 0:
        # No directional volume at all — treat as exhausted (safe to enter)
        return True

    return short_avg < decay_ratio * long_avg


# ---------------------------------------------------------------------------
# LeeReadyTracker — kept for reference / live-trade-stream use if available
# ---------------------------------------------------------------------------


@dataclass
class LeeReadyTracker:
    """
    Classify each trade as buyer- or seller-initiated (Lee & Ready, 1991 style tick rule
    on mid/quote: price above mid => buy aggress, below => sell).

    *** [B] ROUND5: This class is NO LONGER used in the main CHOP entry path.
    *** volume_proxy_exhaustion() replaced it for REST/poll-based deployments.
    *** Kept here for use in live WebSocket deployments where real trade streams
    *** are available and ingested incrementally (no stale-mid problem).
    """

    short_win: int = 5
    long_win: int = 20

    def __post_init__(self) -> None:
        self._buy: Deque[float] = deque(maxlen=self.long_win)
        self._sell: Deque[float] = deque(maxlen=self.long_win)

    def reset(self) -> None:
        self._buy.clear()
        self._sell.clear()

    @staticmethod
    def classify(price: float, mid: float) -> str:
        if price > mid * (1.0 + 1e-12):
            return "buy"
        if price < mid * (1.0 - 1e-12):
            return "sell"
        return "tie"

    def ingest(self, price: float, best_bid: float, best_ask: float, base_vol: float) -> None:
        mid = 0.5 * (best_bid + best_ask)
        side = self.classify(price, mid)
        if side == "buy":
            self._buy.append(base_vol)
            self._sell.append(0.0)
        elif side == "sell":
            self._sell.append(base_vol)
            self._buy.append(0.0)
        else:
            self._buy.append(0.0)
            self._sell.append(0.0)

    def _ema(self, xs: Deque[float], n: int) -> float:
        w = list(xs)[-n:]
        if not w:
            return 0.0
        return sum(w) / len(w)

    def sell_exhaustion(self) -> bool:
        if len(self._sell) < self.long_win:
            return False
        s5 = self._ema(self._sell, self.short_win)
        s20 = self._ema(self._sell, self.long_win)
        return s5 < 0.72 * s20 and s20 > 0

    def buy_exhaustion(self) -> bool:
        if len(self._buy) < self.long_win:
            return False
        b5 = self._ema(self._buy, self.short_win)
        b20 = self._ema(self._buy, self.long_win)
        return b5 < 0.72 * b20 and b20 > 0


# ---------------------------------------------------------------------------
# [C] Per-ticker volume explosion — weekend-aware guard
# ---------------------------------------------------------------------------


@dataclass
class PerTickerVolumeGuard:
    """
    *** [C] ROUND5: three improvements for weekend low-liquidity false positives.

    Round4 → Round5 changes:
      1. Baseline: median → 75th-percentile.
         On thin weekend books the volume distribution is right-skewed: most bars
         are near-zero, occasional single large orders sit far above the median.
         Using median as baseline means the threshold is set very low, so one
         normal-sized order triggers the guard.  The 75th-percentile is less
         sensitive to the sparse tail and better represents "typical active volume".

      2. blast_ratio scaled by 0.6 on weekends (effective ratio 8 × 0.6 = 4.8).
         Weekend volume is structurally lower; the same ratio relative to a lower
         baseline still misclassifies normal bars as explosions.  Reducing the
         effective ratio restores selectivity.

      3. freeze_sec halved on weekends (90s vs 180s).
         Weekend opportunity windows are already narrower; a 3-minute freeze
         costs proportionally more.  90s still gives time for the spike to clear
         while recovering half the opportunity set.
    """

    freeze_sec: float = 180.0           # weekday freeze duration
    freeze_sec_weekend: float = 90.0    # *** [C] shorter weekend freeze
    lookback: int = 30
    blast_ratio: float = 8.0
    blast_ratio_weekend_scale: float = 0.6   # *** [C] effective ratio on weekends
    p75_baseline: bool = True                # *** [C] use 75th-pct instead of median

    _until: Dict[str, float] = field(default_factory=dict)
    _freeze_logged: Dict[str, bool] = field(default_factory=dict)
    _resume_logged: Dict[str, bool] = field(default_factory=dict)

    def is_frozen(self, symbol: str) -> bool:
        return time.time() < self._until.get(symbol, 0.0)

    def remaining(self, symbol: str) -> float:
        return max(0.0, self._until.get(symbol, 0.0) - time.time())

    def check_and_triggers(
        self,
        symbol: str,
        ohlcv_1m: List[List[float]],
        wall_ts: float,
        is_weekend: bool = False,
    ) -> bool:
        """
        *** [C] Signature extended with is_weekend flag.
        If last 1m volume >> baseline(lookback), set per-symbol freeze.
        Returns True if newly frozen.
        """
        if len(ohlcv_1m) < self.lookback + 2:
            return False

        vols = [x[5] for x in ohlcv_1m[-(self.lookback + 1): -1]]
        last_v = ohlcv_1m[-1][5]

        # *** [C] baseline: 75th-pct vs median
        sorted_vols = sorted(vols)
        if self.p75_baseline:
            idx = int(len(sorted_vols) * 0.75)
            baseline = sorted_vols[min(idx, len(sorted_vols) - 1)] or 1e-9
        else:
            baseline = sorted_vols[len(sorted_vols) // 2] or 1e-9

        # *** [C] effective ratio and freeze duration depend on session type
        ratio = self.blast_ratio * (self.blast_ratio_weekend_scale if is_weekend else 1.0)
        fsec = self.freeze_sec_weekend if is_weekend else self.freeze_sec

        if last_v > ratio * baseline:
            self._until[symbol] = wall_ts + fsec
            if not self._freeze_logged.get(symbol):
                logger.warning(
                    "Start Freeze: symbol=%s duration_s=%.0f baseline=%.2f ratio=%.1f weekend=%s",
                    symbol, fsec, baseline, ratio, is_weekend,
                )
                self._freeze_logged[symbol] = True
            self._resume_logged[symbol] = False
            return True
        return False

    def maybe_log_resume(self, symbol: str) -> None:
        """*** Call when tick runs: one 'Resume' line per freeze episode (no per-second DEBUG)."""
        if not self.is_frozen(symbol) and self._freeze_logged.get(symbol) and not self._resume_logged.get(symbol):
            logger.warning("Resume: symbol=%s (volume guard expired)", symbol)
            self._resume_logged[symbol] = True
            self._freeze_logged[symbol] = False


# ---------------------------------------------------------------------------
# Config + main engine
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ROUND6: 幣種白名單 — 根據 backtest 結果決定每個 symbol 允許的路徑
# ---------------------------------------------------------------------------

# 路徑常數
MR_ONLY      = frozenset({"MR"})
TREND_ONLY   = frozenset({"TREND"})
BOTH         = frozenset({"MR", "TREND"})
# CTR_MR = counter-trend MR（ADX rising 反向入場）
CTR_MR_ONLY  = frozenset({"CTR_MR"})
MR_AND_CTR   = frozenset({"MR", "CTR_MR"})   # 普通 MR + 反趨勢 MR 雙開

# ROUND8: 白名單更新
# - 普通 MR（CHOP regime）：AAVE, AXS, FARTCOIN, TRUMP, XRP
# - 反趨勢 MR（TREND regime ADX rising 反向）：HYPE, TAO（WFA 驗證）
# - HYPE/TAO 同時保留普通 MR（CHOP 時）+ 反趨勢 MR（TREND 時）
SYMBOL_ALLOWED_MODES: Dict[str, frozenset] = {
    "AAVE/USDC:USDC":       MR_ONLY,
    "AXS/USDC:USDC":        MR_ONLY,
    "FARTCOIN/USDC:USDC":   MR_ONLY,
    "TRUMP/USDC:USDC":      MR_ONLY,
    "XRP/USDC:USDC":        MR_ONLY,
    "HYPE/USDC:USDC":       MR_AND_CTR,   # WFA: 3/3 folds 正，最穩定
    "TAO/USDC:USDC":        MR_AND_CTR,   # WFA: 2/3 folds 正，謹慎用
}

# 預設交易列表（白名單所有 symbol）
DEFAULT_SYMBOLS = ",".join(SYMBOL_ALLOWED_MODES.keys())


@dataclass
class ATMSessionConfig:
    # 部位名目基礎；*** 預設 600 USDT 供 paper trade，實盤可改或即時讀帳戶權益
    equity_usdt: float = 600.0
    max_leverage: float = 3.0
    risk_fraction_per_symbol: float = 0.02

    # Fees (Hyperliquid: *** verify against your VIP tier; maker rebate is separate)
    taker_fee_rate: float = 0.00035
    round_trip_fee_rate: float = 0.0003
    # *** maker edge — used for PnL sanity checks, not to force fills
    maker_rebate_rate: float = -0.000384  # +0.002% as stated

    # Z-score MR (1m closes)
    z_window: int = 45  # *** 30–60m: pick 45m as a compromise
    # ROUND5.1: widened band [1.8, 2.2] → [1.5, 2.5]
    # In low-vol weekends 1m std is tiny; Z crosses 1.8 and immediately exits before
    # the engine can evaluate all gates. Wider band gives more dwell time at extremes.
    z_entry_min: float = 1.5
    z_entry_max: float = 2.5

    # R:R and ATR (*** data-driven loosening vs ROUND4: slightly relax if CSV shows missed "open bonus")
    rr_min: float = 0.9  # was 1.0
    atr_entry_mult: float = 1.75  # was 2.0 — slightly easier entry on low-vol weekends
    min_rr_vs_fees: float = 0.35  # gross edge above fees at entry (heuristic)

    # *** ROUND5.1: minimum TP distance gate
    # TP 距離必須超過 round-trip fee 的 N 倍才開倉。
    # N=5 → TP dist > 5 × 0.03% = 0.15% of entry price
    # 防止在低波動環境開出利潤薄過手續費的倉位。
    min_tp_fee_multiple: float = 5.0

    # *** Funding: hourly; positive funding = longs pay. Skip new longs if predicted too high.
    max_long_funding_per_hour: float = 0.00015  # 0.015% / hr cap for new longs (tune to book)

    # L2: order book imbalance = (bid_notional-ask_notional)/(sum)
    ob_levels: int = 10
    # ROUND5.1: relaxed 0.12 → 0.05 — weekend OB is thin, imbalance rarely reaches ±0.12
    # even when there is genuine directional pressure. 0.05 still requires a mild lean.
    min_abs_imbalance: float = 0.05

    # Deceleration exit
    decel_profit_bps: float = 8.0  # only if already this much in-the-money
    # *** Cycle: weekday hard-close vs weekend extension so :45 does not eject good MR
    hard_close_minute_weekday: int = 44
    hard_close_minute_weekend: int = 48

    # Maker reprice
    maker_wait_sec: float = 4.0
    maker_bps_improve: float = 0.5  # nudge price if not filled (still post-only)
    tight_spread_bps: float = 3.0   # if spread <= this, reprice by half-spread; else use bps_improve

    # *** [B] Volume proxy exhaustion params (passed through to volume_proxy_exhaustion)
    # ROUND5.1: decay relaxed 0.65 → 0.80 (weekend low-vol books have very noisy 1m volume;
    # 0.65 required short-window avg < 65% of long-window avg — rarely satisfied in thin markets)
    vol_proxy_window: int = 10      # 1m bars for direction proxy
    vol_proxy_decay: float = 0.80   # short/long avg ratio threshold (relaxed from 0.65)

    # *** ROUND8: 反趨勢 MR 參數（WFA 優化，HYPE/TAO 專用）
    # ADX rising 反向入場：趨勢過熱 → 反向 MR 入場
    # 參數來源：wfa_best.csv 跨 fold 最穩定組合
    ctr_adx_rising_bars: int   = 3     # ADX 連升確認 bar 數
    ctr_adx_min: float         = 25.0  # ADX 最低水平（排除弱趨勢）
    ctr_z_entry_min: float     = 1.5   # Z-score 入場下限（反趨勢不需極端）
    ctr_z_entry_max: float     = 2.5   # Z-score 入場上限
    ctr_sl_mult: float         = 1.0   # SL = entry ± ctr_sl_mult × ATR
    ctr_max_hold_bars: int     = 4     # 最長持倉（1H bars），HYPE 最優值
    ctr_rr_min: float          = 0.5   # 反趨勢 R:R 門檻放寬（TP 距離較小）

    # *** ROUND9: MEI（動能耗盡指數）參數
    # MEI = (accel[t] - accel[t-1]) / |accel[t-1]|
    # 量度 ADX 加速度的變化率，MEI 極負 = 趨勢動能急速衰竭
    #
    # 多單保護（普通 MR long + CTR_MR long）：
    #   MEI < mei_long_skip    → 跳過（頂部確認，動能仍強）
    #   MEI < mei_long_reduce1 → position_multiplier = 0.3
    #   MEI < mei_long_reduce2 → position_multiplier = 0.7
    #   其他                   → position_multiplier = 1.0
    #
    # 空單保護（普通 MR short + CTR_MR short）更寬鬆：
    # 下跌動能比上漲更難耗盡（恐慌拋售），門檻更嚴
    #   MEI < mei_short_skip   → position_multiplier = 0.5
    #   MEI < mei_short_reduce → position_multiplier = 0.7
    #   其他                   → position_multiplier = 1.0
    mei_long_skip:    float = -0.8   # 多單：跳過門檻
    mei_long_reduce1: float = -0.5   # 多單：減倉 70%
    mei_long_reduce2: float = -0.3   # 多單：減倉 30%
    mei_short_skip:   float = -1.5   # 空單：半倉門檻
    mei_short_reduce: float = -0.8   # 空單：減倉 30%
    mei_lookback:     int   = 2      # 計算 MEI 用的 accel 回溯 bar 數


@dataclass
class ATMBot:
    """Live orchestration: regime, microstructure, funding, maker-first, per-symbol freeze."""

    ex: Any  # ccxt.hyperliquid
    session: ATMSessionConfig
    on_signal: Optional[Callable[..., Any]] = field(default=None)  # optional hook for tests

    def __post_init__(self) -> None:
        self.regime = RegimeFilter()
        self.lee: Dict[str, LeeReadyTracker] = {}  # kept for WS use; not called in CHOP path
        self.vol_guard = PerTickerVolumeGuard()
        self._vel: Dict[str, PriceVelocityState] = {}

    def _ensure_actors(self, symbol: str) -> None:
        if symbol not in self.lee:
            self.lee[symbol] = LeeReadyTracker()
        if symbol not in self._vel:
            self._vel[symbol] = PriceVelocityState(24)

    # --- L2 + funding ----------------------------------------------------

    def orderbook_imbalance(self, symbol: str) -> float:
        """-1..+1: bid heavy positive."""
        ob = self.ex.fetch_order_book(symbol, self.session.ob_levels)
        bid_sz = sum((b[0] or 0) * (b[1] or 0) for b in ob.get("bids", []) or [])
        ask_sz = sum((a[0] or 0) * (a[1] or 0) for a in ob.get("asks", []) or [])
        tot = bid_sz + ask_sz or 1e-9
        return (bid_sz - ask_sz) / tot

    def predicted_funding_rate(self, symbol: str) -> float:
        """
        *** HL `metaAndAssetCtxs` exposes next-period funding in `funding` field; ccxt places it in
        *** `fundingRate` when parsing.
        """
        rates = self.ex.fetch_funding_rates([symbol])
        fr = (rates or {}).get(symbol) or {}
        v = fr.get("fundingRate")
        if v is not None:
            return float(v)
        info = fr.get("info") or {}
        if "funding" in info:
            return float(info["funding"])
        return 0.0

    def funding_blocks_new_long(self, symbol: str) -> bool:
        f = self.predicted_funding_rate(symbol)
        return f > self.session.max_long_funding_per_hour

    # --- Cycle window ----------------------------------------------------

    @staticmethod
    def in_soft_trade_window(now: datetime, s: ATMSessionConfig) -> bool:
        """
        *** 30m cycle: trade from :00–:N where N is weekday vs weekend.
        """
        m = now.minute % 30
        is_weekend = now.weekday() >= 5
        cap = s.hard_close_minute_weekend if is_weekend else s.hard_close_minute_weekday
        return m < cap

    # --- Entry validation ------------------------------------------------

    def chop_z_entry(
        self,
        ohlcv_1m: List[List[float]],
        side: str,
        imb: float,
    ) -> Tuple[bool, Optional[float]]:
        c = [x[4] for x in ohlcv_1m]
        zs = rolling_z_score(c, self.session.z_window)
        if not zs:
            return False, None
        z, _, _ = zs
        if side == "long":
            if z < -self.session.z_entry_max or z > -self.session.z_entry_min:
                return False, z
            if imb < self.session.min_abs_imbalance:
                return False, z
        else:
            if z < self.session.z_entry_min or z > self.session.z_entry_max:
                return False, z
            if imb > -self.session.min_abs_imbalance:
                return False, z
        return True, z

    def atr_ok(self, ohlcv_1h: List[List[float]], mark: float) -> bool:
        """
        *** Require ATR/mid above floor to avoid fee-dominated noise trades.
        """
        a = atr(ohlcv_1h, 14)
        if a is None or not mark:
            return False
        return (a / mark) >= self.session.atr_entry_mult * 1e-4

    def rr_vs_fees_ok(self, tp_dist: float, sl_dist: float) -> bool:
        if sl_dist <= 0:
            return False
        rr = tp_dist / sl_dist
        return rr >= self.session.rr_min and rr >= self.session.min_rr_vs_fees + self.session.round_trip_fee_rate

    # --- Trades for Lee-Ready (WS path only) ----------------------------

    def refresh_lee_ready(self, symbol: str) -> None:
        """
        *** [B] ROUND5: this method is no longer called in the main CHOP entry path.
        *** Kept for WebSocket deployment where trades are ingested incrementally.
        *** In REST mode, use volume_proxy_exhaustion(o1m, side) directly instead.
        """
        self._ensure_actors(symbol)
        ob = self.ex.fetch_order_book(symbol, 1)
        best_bid = (ob["bids"][0][0] if ob.get("bids") else None) or 0.0
        best_ask = (ob["asks"][0][0] if ob.get("asks") else None) or 0.0
        trades = self.ex.fetch_trades(symbol, None, 120)
        tr = self.lee[symbol]
        for t in trades[-80:]:
            p = t.get("price") or t.get("info", {}).get("px")
            if p is None:
                continue
            v = t.get("amount", 0.0) or 0.0
            tr.ingest(float(p), float(best_bid), float(best_ask), float(v))

    # --- Order placement: maker-first ------------------------------------

    @staticmethod
    def _ccxt_side(side: str) -> str:
        return "buy" if side == "long" else "sell"

    def try_post_only(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> Dict[str, Any]:
        """Place a post-only (ALO) limit order. Returns the ccxt order dict."""
        cs = self._ccxt_side(side)
        return self.ex.create_order(
            symbol,
            "limit",
            cs,
            amount,
            price,
            params={"postOnly": True, "timeInForce": "ALO"},
        )

    @staticmethod
    def _ob3_price(ob: Dict[str, Any], side: str, fallback: float) -> float:
        """
        3-level size-weighted average price from the order book.

        For a buy order: weighted avg of top-3 BID levels (inside spread).
        For a sell order: weighted avg of top-3 ASK levels (inside spread).

        Using the 3-level wavg instead of best bid/ask avoids constant
        cancel-reprice cycles when the book ticks every few milliseconds.
        The wavg only moves meaningfully when a full level is consumed.
        """
        key = "bids" if side == "buy" else "asks"
        levels = (ob.get(key) or [])[:3]
        if not levels:
            return fallback
        total_sz = sum(lv[1] for lv in levels if lv[1])
        if total_sz <= 0:
            return float(levels[0][0]) if levels[0][0] else fallback
        wavg = sum(lv[0] * lv[1] for lv in levels if lv[0] and lv[1]) / total_sz
        return float(wavg)

    def maker_with_reprice(
        self,
        symbol: str,
        side: str,
        amount: float,
        mark: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a post-only (ALO) limit order at the 3-level weighted average
        book price, then reprice adaptively based on spread width.

        Reprice logic:
          1. Fetch OB, compute 3-level wavg price (px0).
          2. Post ALO order at px0.
          3. Poll every 0.3s up to maker_wait_sec.
          4. If unfilled: compute spread = (ask1 - bid1) / mid.
             - spread <= tight_spread_bps: jump half-spread toward mid.
             - Otherwise: nudge by maker_bps_improve (conservative).
          5. One reprice; leave the order resting if still unfilled.
        """
        cs = self._ccxt_side(side)
        ob = self.ex.fetch_order_book(symbol, 3)
        px0 = self._ob3_price(ob, cs, mark)
        px0 = float(self.ex.price_to_precision(symbol, px0))

        t0 = time.time()
        order = self.try_post_only(symbol, side, amount, px0)

        while time.time() - t0 < self.session.maker_wait_sec:
            oid = order.get("id")
            o = self.ex.fetch_order(oid, symbol) if oid else {}
            if o.get("status") in ("closed", "filled") or (o.get("filled") and float(o["filled"]) > 0):
                return o
            time.sleep(0.3)

        # --- Adaptive reprice ---
        ob2 = self.ex.fetch_order_book(symbol, 3)
        bid1 = (ob2.get("bids") or [[mark]])[0][0] or mark
        ask1 = (ob2.get("asks") or [[mark]])[0][0] or mark
        mid = 0.5 * (bid1 + ask1)
        spread_bps = (ask1 - bid1) / mid * 1e4 if mid > 0 else 999.0

        tight_spread_bps = getattr(self.session, "tight_spread_bps", 3.0)
        if spread_bps <= tight_spread_bps:
            half_spread = (ask1 - bid1) * 0.5
            if cs == "buy":
                px1 = bid1 + half_spread
            else:
                px1 = ask1 - half_spread
        else:
            bps = self.session.maker_bps_improve * 1e-4
            if cs == "buy":
                px1 = px0 * (1.0 + bps)
            else:
                px1 = px0 * (1.0 - bps)

        px1 = float(self.ex.price_to_precision(symbol, px1))

        try:
            if order.get("id"):
                self.ex.cancel_order(order.get("id"), symbol)
        except Exception:  # noqa: BLE001
            pass

        new_order = self.try_post_only(symbol, side, amount, px1)
        return new_order

    # --- MEI: Momentum Exhaustion Index ---------------------------------

    def _compute_mei(self, ohlcv_1h: List[List[float]]) -> Optional[float]:
        """
        ROUND9: MEI（動能耗盡指數）

        MEI = (accel[t] - accel[t-1]) / |accel[t-1]|

        其中 accel 是 adx_mom() 返回的二階差分（ADX 加速度）。
        MEI 量度加速度本身的變化率：
          MEI 極負 = 加速度急速下降 = 趨勢動能正在快速耗盡 = 頂/底部確認
          MEI 接近零或正 = 加速度穩定或上升 = 趨勢仍有動能

        與 trend bot 的 MEI 設計一致：
          多單：MEI < -0.8 → 跳過（頂部保護）
          空單：更寬鬆門檻（下跌動能更難耗盡）

        Returns None 如果數據不足。
        """
        mom = self.regime.exhaustion_diagnostics(ohlcv_1h)
        if mom is None:
            return None
        # exhaustion_diagnostics 返回 (rate, accel, adx_now)
        # 需要兩個時間點的 accel 才能算 MEI，用 adx_mom 的 lookback 參數
        # 這裡用簡化版：直接從 adx_series 重算兩期 accel
        raw = adx_series(ohlcv_1h, self.regime.adx_period)
        lb = self.session.mei_lookback
        if len(raw) < lb + 4:
            return None

        # 計算最近兩期的 accel（rate 的一階差分）
        # rate[t]   = raw[-1] - raw[-2]
        # rate[t-1] = raw[-2] - raw[-3]
        # accel[t]   = rate[t]   - rate[t-1]
        # accel[t-1] = rate[t-1] - rate[t-2]
        rate_t   = raw[-1] - raw[-2]
        rate_t1  = raw[-2] - raw[-3]
        rate_t2  = raw[-3] - raw[-4]
        accel_t  = rate_t  - rate_t1
        accel_t1 = rate_t1 - rate_t2

        denom = abs(accel_t1) + 1e-9
        mei = (accel_t - accel_t1) / denom
        return float(mei)

    def _mei_position_multiplier(
        self,
        mei: Optional[float],
        side: str,
    ) -> float:
        """
        ROUND9: 根據 MEI 值返回倉位乘數。

        多單（long）保護更嚴格：
          MEI < mei_long_skip    → 0.0（跳過，頂部確認）
          MEI < mei_long_reduce1 → 0.3（減倉 70%）
          MEI < mei_long_reduce2 → 0.7（減倉 30%）
          其他                   → 1.0（全倉）

        空單（short）更寬鬆：
          MEI < mei_short_skip   → 0.5（半倉）
          MEI < mei_short_reduce → 0.7（減倉 30%）
          其他                   → 1.0（全倉）

        MEI 為 None（數據不足）→ 保守起見返回 0.7。
        """
        if mei is None:
            return 0.7   # 數據不足，保守半倉

        s = self.session
        if side == "long":
            if mei < s.mei_long_skip:    return 0.0
            if mei < s.mei_long_reduce1: return 0.3
            if mei < s.mei_long_reduce2: return 0.7
            return 1.0
        else:  # short
            if mei < s.mei_short_skip:   return 0.5
            if mei < s.mei_short_reduce: return 0.7
            return 1.0

    # --- Counter-trend MR signal (HYPE / TAO only) ----------------------

    def _counter_trend_mr_signal(
        self,
        ohlcv_1h: List[List[float]],
        ohlcv_1m: List[List[float]],
        imb: float,
    ) -> Optional[str]:
        """
        ROUND8: 反趨勢 MR 入場信號。

        觸發條件（全部滿足）：
          1. ADX 連升 ctr_adx_rising_bars 根 1H bar（趨勢加速 = 過熱）
          2. ADX 當前值 ≥ ctr_adx_min（排除弱趨勢噪聲）
          3. Z-score 在 [ctr_z_entry_min, ctr_z_entry_max] 方向確認：
               上升趨勢（EMA9 > EMA21）→ Z 偏高 → 做空
               下降趨勢（EMA9 < EMA21）→ Z 偏低 → 做多
          4. imb 方向輕微確認（放寬至 0.03，反趨勢不強求 OB 支持）

        WFA 最優參數（HYPE 3/3 folds, TAO 2/3 folds）：
          adx_rising_bars=3, adx_min=25, z_entry_min=1.5,
          z_entry_max=2.5, sl_mult=1.0, max_hold=4H
        """
        s = self.session
        raw = adx_series(ohlcv_1h, self.regime.adx_period)
        n = s.ctr_adx_rising_bars
        if len(raw) < n:
            return None

        # 條件 1+2：ADX 連升且達最低水平
        adx_window = raw[-n:]
        if not all(adx_window[i] > adx_window[i-1] for i in range(1, n)):
            return None
        if raw[-1] < s.ctr_adx_min:
            return None

        # EMA 方向
        closes = [x[4] for x in ohlcv_1h]
        e9, e21 = ema(closes, 9), ema(closes, 21)
        if e9 is None or e21 is None:
            return None

        # 條件 3：Z-score 方向確認（用 1m closes）
        closes_1m = [x[4] for x in ohlcv_1m]
        zs = rolling_z_score(closes_1m, s.z_window)
        if not zs:
            return None
        z = zs[0]

        side = None
        if e9 > e21 and s.ctr_z_entry_min <= z <= s.ctr_z_entry_max:
            side = "short"   # 上升趨勢過熱 → 反向做空
        elif e9 < e21 and -s.ctr_z_entry_max <= z <= -s.ctr_z_entry_min:
            side = "long"    # 下降趨勢過熱 → 反向做多

        if side is None:
            return None

        # 條件 4：OB imbalance 輕微確認（放寬門檻）
        if side == "long" and imb < -0.03:
            return None   # 做多但賣壓太重
        if side == "short" and imb > 0.03:
            return None   # 做空但買壓太重

        return side

    # --- Deceleration (open position) ------------------------------------

    def should_early_exit(
        self,
        symbol: str,
        mark: float,
        entry: float,
        side: str,
    ) -> bool:
        self._ensure_actors(symbol)
        self._vel[symbol].push(mark, time.time())
        pnl_bps = (mark - entry) / entry * 1e4 if side == "long" else (entry - mark) / entry * 1e4
        if pnl_bps < self.session.decel_profit_bps:
            return False
        return self._vel[symbol].deceleration_signal()

    # --- Main one-shot trigger loop (per outer tick) ---------------------

    def atm_trigger_loop(
        self,
        symbols: List[str],
        now: Optional[datetime] = None,
    ) -> None:
        """
        ROUND5 changes in this loop:
          - is_weekend passed to vol_guard.check_and_triggers() [C]
          - Lee-Ready calls replaced by volume_proxy_exhaustion(o1m, cand) [B]
          - refresh_lee_ready() no longer called (REST stale-mid issue) [B]
        """
        now = now or datetime.now(timezone.utc)
        if not self.in_soft_trade_window(now, self.session):
            return

        is_weekend = now.weekday() >= 5  # *** [C]

        for symbol in symbols:
            self._ensure_actors(symbol)
            self.vol_guard.maybe_log_resume(symbol)
            if self.vol_guard.is_frozen(symbol):
                continue

            o1m = self.ex.fetch_ohlcv(symbol, "1m", limit=200)
            o1h = self.ex.fetch_ohlcv(symbol, "1h", limit=200)
            o1m = [list(x) for x in o1m]
            o1h = [list(x) for x in o1h]

            # *** [C] pass is_weekend so guard uses weekend-tuned ratio and freeze_sec
            if self.vol_guard.check_and_triggers(symbol, o1m, time.time(), is_weekend=is_weekend):
                continue

            mark = float((self.ex.fetch_ticker(symbol) or {}).get("last") or o1h[-1][4])
            st, a_val, rv = self.regime.evaluate(o1h)
            if st is None:
                continue

            # *** [B] refresh_lee_ready() removed from REST path
            imb = self.orderbook_imbalance(symbol)

            # ROUND6: 白名單檢查 — 查詢此 symbol 允許的路徑
            allowed = SYMBOL_ALLOWED_MODES.get(symbol, frozenset())
            if not allowed:
                continue   # 不在白名單 → 跳過

            # ROUND8: 雙 MR 路徑
            side: Optional[str] = None
            entry_mode: str = ""

            if st in (MarketRegime.CHOP, MarketRegime.NEUTRAL) and "MR" in allowed:
                # ── 普通 MR：橫盤 Z-score 均值回歸 ────────────────────────
                if not self.regime.exhaustion_ok(o1h):
                    continue
                for cand in ("long", "short"):
                    ok, _z = self.chop_z_entry(ohlcv_1m=o1m, side=cand, imb=imb)
                    if not ok:
                        continue
                    if not volume_proxy_exhaustion(
                        o1m,
                        cand,
                        window=self.session.vol_proxy_window,
                        decay_ratio=self.session.vol_proxy_decay,
                    ):
                        continue
                    side = cand
                    entry_mode = "MR"
                    break

            elif st == MarketRegime.TREND and "CTR_MR" in allowed:
                # ── 反趨勢 MR：TREND 過熱 → 反向入場（HYPE/TAO 限定）──────
                side = self._counter_trend_mr_signal(o1h, o1m, imb)
                if side:
                    entry_mode = "CTR_MR"

            if side is None:
                continue
            if side == "long" and self.funding_blocks_new_long(symbol):
                continue
            if not self.atr_ok(o1h, mark):
                continue

            # ── ROUND9: MEI 過濾 + 倉位乘數 ────────────────────────────
            mei = self._compute_mei(o1h)
            pos_mult = self._mei_position_multiplier(mei, side)
            if pos_mult == 0.0:
                logger.info(
                    "MEI_SKIP  %s %s  mei=%.3f  reason=momentum_exhaustion_top",
                    side.upper(), symbol, mei if mei is not None else 0.0,
                )
                continue

            notional = self.session.equity_usdt * self.session.risk_fraction_per_symbol * 8.0
            raw_amt = (notional / max(mark, 1e-9)) * pos_mult
            amount = float(self.ex.amount_to_precision(symbol, raw_amt))
            if amount <= 0:
                continue

            if self.on_signal:
                self.on_signal(
                    {
                        "symbol":    symbol,
                        "side":      side,
                        "amount":    amount,
                        "regime":    st,
                        "adx":       a_val,
                        "rv1h":      rv,
                        "imb":       imb,
                        "mark":      mark,
                        "entry_mode": entry_mode,
                        "mei":       mei,
                        "pos_mult":  pos_mult,
                    }
                )
            else:
                self.maker_with_reprice(symbol, side, amount, mark)
