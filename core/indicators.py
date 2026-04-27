"""
Pure numeric indicators (OHLCV rows: [ms, o, h, l, c, v]).
No side effects; unit-test friendly.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

Num = float


def ema(values: Sequence[Num], period: int) -> Optional[Num]:
    """EMA at end of `values` with SMA(period) seed (needs len >= period)."""
    if len(values) < period or period < 1:
        return None
    k = 2.0 / (period + 1)
    e = sum(values[:period]) / float(period)
    for v in values[period:]:
        e = v * k + e * (1.0 - k)
    return e


def ema_series(closes: Sequence[Num], period: int) -> List[Num]:
    """SMA-seeded EMA values from bar index `period-1` through end of `closes`."""
    if len(closes) < period or period < 1:
        return []
    k = 2.0 / (period + 1)
    e = sum(closes[:period]) / float(period)
    out: List[Num] = [e]
    for i in range(period, len(closes)):
        e = closes[i] * k + e * (1.0 - k)
        out.append(e)
    return out


def true_range(ohlcv: List[List[Num]], i: int) -> float:
    _, o, h, l, c, _ = ohlcv[i]
    if i == 0:
        return h - l
    pc = ohlcv[i - 1][4]
    return max(h - l, abs(h - pc), abs(l - pc))


def atr(ohlcv: List[List[Num]], period: int) -> Optional[float]:
    if len(ohlcv) < period + 1:
        return None
    trs = [true_range(ohlcv, i) for i in range(1, len(ohlcv))]
    w = trs[-period:]
    return sum(w) / len(w)


def adx_wilder_full(ohlcv: List[List[Num]], period: int) -> Optional[float]:
    """Last-bar ADX(period) with Wilder smoothing of +DI, -DI, and DX."""
    s = adx_series(ohlcv, period)
    return s[-1] if s else None


def adx_series(ohlcv: List[List[Num]], period: int) -> List[float]:
    """
    Full ADX time-series (same Wilder smoothing as adx_wilder_full).
    Returns one value per bar from index (3*period - 1) onward.
    Needed by adx_momentum so we can diff the series without recomputing.
    """
    n = len(ohlcv)
    if n < period * 3:
        return []
    trs, pdm, mdm = [], [], []
    for i in range(1, n):
        trs.append(true_range(ohlcv, i))
        h0, l0 = ohlcv[i - 1][2], ohlcv[i - 1][3]
        h1, l1 = ohlcv[i][2], ohlcv[i][3]
        up = h1 - h0
        down = l0 - l1
        pdm.append(up if (up > down and up > 0) else 0.0)
        mdm.append(down if (down > up and down > 0) else 0.0)

    def w_smooth(p: int, xs: List[float]) -> List[float]:
        if len(xs) < p:
            return []
        out = [sum(xs[:p])]
        for j in range(p, len(xs)):
            out.append((out[-1] * (p - 1) + xs[j]) / p)
        return out

    tr_s = w_smooth(period, trs)
    p_s = w_smooth(period, pdm)
    m_s = w_smooth(period, mdm)
    if not tr_s or len(tr_s) < 2:
        return []
    dxs: List[float] = []
    m = min(len(tr_s), len(p_s), len(m_s))
    for j in range(m):
        tr = tr_s[j] or 1e-12
        pdi = 100.0 * p_s[j] / tr
        mdi = 100.0 * m_s[j] / tr
        den = abs(pdi + mdi) or 1e-12
        dxs.append(100.0 * abs(pdi - mdi) / den)
    if len(dxs) < period:
        return []
    return w_smooth(period, dxs)


def adx_momentum(
    ohlcv: List[List[Num]],
    period: int = 14,
    smooth: int = 3,
    lookback: int = 4,
) -> Optional[Tuple[float, float, float]]:
    """
    Measure ADX momentum via smoothed first-difference and second-difference.

    Why smooth before differencing
    --------------------------------
    ADX is already a double-Wilder-smoothed value (lag ≈ 2*period bars).
    Raw bar-to-bar differences on ADX amplify the residual tick noise that
    survives Wilder smoothing, especially when ADX < 20 where the denominator
    (|+DI| + |-DI|) is small and unstable.  A short EMA(smooth=3) on the ADX
    series suppresses that noise before we diff, at the cost of only 1-2 extra
    bars of lag — acceptable because the signal we care about (exhaustion) is
    slow by nature.

    Parameters
    ----------
    period   : ADX period (default 14, must match RegimeFilter.adx_period)
    smooth   : EMA window applied to raw ADX series before differencing (3–5)
    lookback : how many smoothed bars to average for rate/diff(rate) (3–5)

    Returns
    -------
    (rate, accel, adx_now) where:
      rate      = mean of last `lookback` first-differences of smoothed ADX
                  positive → ADX still rising (trend building)
                  negative → ADX falling (momentum ebbing)
      accel     = mean of last `lookback` second-differences (diff of rate)
                  negative → rate itself is declining (deceleration confirmed)
                  positive → rate is recovering (do not call exhaustion yet)
      adx_now   = last smoothed ADX value (for level check, e.g. < 20)

    Exhaustion signal (caller's responsibility to combine with Z and Lee-Ready):
      rate < 0  AND  accel < 0  AND  adx_now < adx_chop_threshold
    """
    raw = adx_series(ohlcv, period)
    # Need enough bars: smooth seed + lookback for both rate and accel
    min_len = smooth + lookback + 2
    if len(raw) < min_len:
        return None

    # Step 1: EMA-smooth the raw ADX series to kill tick noise
    smoothed = ema_series(raw, smooth)          # len = len(raw) - smooth + 1
    if len(smoothed) < lookback + 2:
        return None

    # Step 2: first-difference on smoothed series  (rate of ADX)
    d1 = [smoothed[i] - smoothed[i - 1] for i in range(1, len(smoothed))]
    if len(d1) < lookback + 1:
        return None

    # Step 3: second-difference  (acceleration / diff of rate)
    d2 = [d1[i] - d1[i - 1] for i in range(1, len(d1))]
    if len(d2) < lookback:
        return None

    rate = sum(d1[-lookback:]) / lookback
    accel = sum(d2[-lookback:]) / lookback
    adx_now = smoothed[-1]
    return (rate, accel, adx_now)


def rolling_z_score(closes: Sequence[Num], window: int) -> Optional[Tuple[float, float, float]]:
    """
    Z = (c - mean) / std on last `window` closes. Returns (z, mean, std).
    """
    if len(closes) < window or window < 2:
        return None
    w = list(closes[-window:])
    mu = sum(w) / len(w)
    var = sum((x - mu) ** 2 for x in w) / (len(w) - 1)
    std = math.sqrt(var) if var > 0 else 1e-12
    z = (w[-1] - mu) / std
    return (z, mu, std)


def rolling_return_vol(closes: Sequence[Num], window: int) -> Optional[float]:
    if len(closes) < window + 1 or window < 2:
        return None
    rets = []
    c = list(closes)
    for i in range(len(c) - window, len(c)):
        if i < 1:
            continue
        p0, p1 = c[i - 1], c[i]
        if p0 and p1:
            rets.append((p1 - p0) / p0)
    if len(rets) < 2:
        return None
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


class PriceVelocityState:
    """Tracks return velocity and acceleration for deceleration early-exit logic."""

    def __init__(self, maxlen: int = 20) -> None:
        self._p: Deque[float] = deque(maxlen=maxlen)
        self._t: Deque[float] = deque(maxlen=maxlen)

    def push(self, price: float, wall_time: float) -> None:
        self._p.append(price)
        self._t.append(wall_time)

    def deceleration_signal(self) -> bool:
        """
        True if short-horizon absolute return speed is decaying (stall before TP).
        """
        if len(self._p) < 5:
            return False
        p = list(self._p)
        t = list(self._t)
        dts = max(t[-1] - t[0], 1e-6)
        v_long = (p[-1] - p[0]) / dts
        mid = max(1, len(p) // 2)
        dt1 = max(t[mid] - t[0], 1e-6)
        dt2 = max(t[-1] - t[mid], 1e-6)
        v1 = (p[mid] - p[0]) / dt1
        v2 = (p[-1] - p[mid]) / dt2
        return abs(v2) < 0.45 * abs(v1) and abs(v_long) < 0.55 * (abs((p[mid] - p[0]) / dt1) + 1e-9)


# Aliases
adx = adx_wilder_full          # single scalar — RegimeFilter.evaluate()
adx_mom = adx_momentum         # (rate, accel, adx_now) — RegimeFilter.exhaustion_diagnostics()
# adx_series is already defined above; re-exported here so callers can do:
#   from core.indicators import adx_series
# without hunting for the function name.
# *** ROUND5: used by RegimeFilter.exhaustion_ok() (confirm_bars counter)
#             and paper_trade._log_diagnostics() (exh_bars display)
__all__ = [
    "ema", "ema_series",
    "true_range", "atr",
    "adx_wilder_full", "adx_series", "adx_momentum",
    "adx", "adx_mom",
    "rolling_z_score", "rolling_return_vol",
    "PriceVelocityState",
]
