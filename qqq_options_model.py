#!/usr/bin/env python3
"""
QQQ Options Decision Model
--------------------------
Educational research tool that:

1. Downloads QQQ price history and the available option chain with yfinance.
2. Calculates EMA, SMA, RSI, MACD, Bollinger Bands, ATR and relative volume.
3. Uses Black-Scholes-Merton to estimate theoretical value and Greeks.
4. Estimates expected move and risk-neutral probability of finishing ITM.
5. Builds a transparent technical/catalyst/liquidity score.
6. Ranks option contracts by a user-defined balance of probability, liquidity,
   payoff potential, theta risk and technical alignment.
7. Exports CSV files and PNG charts.
8. Optionally sends a summary of the top contracts to a Telegram chat.

This script does NOT predict market direction or guarantee profitable trades.
Black-Scholes values and probabilities are model estimates, not execution prices
or real-world guarantees.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm

OptionType = Literal["call", "put"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    ticker: str = "SPY"
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.005
    account_size: float = 300.0
    max_risk_pct: float = 0.10
    min_open_interest: int = 100
    min_volume: int = 20
    max_spread_pct: float = 0.20
    target_return: float = 1.00       # 100%
    scenario_move_sigma: float = 1.0
    intraday_period: str = "5d"
    intraday_interval: str = "5m"
    daily_period: str = "1y"
    output_dir: Path = Path("qqq_model_output")
    history_dir: Path = Path("history")
    skew_target_pct: float = 0.08  # ~8% OTM strikes used for the skew comparison


@dataclass(frozen=True)
class BSResult:
    value: float
    delta: float
    gamma: float
    theta_per_day: float
    vega_per_vol_point: float
    probability_itm: float
    d1: float
    d2: float


# ---------------------------------------------------------------------------
# Black-Scholes-Merton
# ---------------------------------------------------------------------------

def black_scholes_merton(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: OptionType,
) -> BSResult:
    """Price a European option and calculate core Greeks.

    QQQ options are American-style, so this is an approximation. For very
    short-dated, near-the-money QQQ options, the approximation is often useful,
    but it can differ from live prices because of bid/ask spreads, changing IV,
    dividends and early-exercise value.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike must be positive.")
    if time_years <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        delta = (
            1.0 if option_type == "call" and spot > strike
            else -1.0 if option_type == "put" and spot < strike
            else 0.0
        )
        return BSResult(intrinsic, delta, 0.0, 0.0, 0.0, float(intrinsic > 0), 0.0, 0.0)
    if volatility <= 0:
        raise ValueError("Volatility must be positive.")

    sqrt_t = math.sqrt(time_years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * time_years
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    discounted_spot = spot * math.exp(-dividend_yield * time_years)
    discounted_strike = strike * math.exp(-rate * time_years)
    pdf_d1 = norm.pdf(d1)

    gamma = math.exp(-dividend_yield * time_years) * pdf_d1 / (
        spot * volatility * sqrt_t
    )
    vega = discounted_spot * pdf_d1 * sqrt_t / 100.0

    if option_type == "call":
        value = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
        delta = math.exp(-dividend_yield * time_years) * norm.cdf(d1)
        theta_annual = (
            -(discounted_spot * pdf_d1 * volatility) / (2.0 * sqrt_t)
            - rate * discounted_strike * norm.cdf(d2)
            + dividend_yield * discounted_spot * norm.cdf(d1)
        )
        probability_itm = norm.cdf(d2)
    else:
        value = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
        delta = math.exp(-dividend_yield * time_years) * (norm.cdf(d1) - 1.0)
        theta_annual = (
            -(discounted_spot * pdf_d1 * volatility) / (2.0 * sqrt_t)
            + rate * discounted_strike * norm.cdf(-d2)
            - dividend_yield * discounted_spot * norm.cdf(-d1)
        )
        probability_itm = norm.cdf(-d2)

    return BSResult(
        value=max(value, 0.0),
        delta=delta,
        gamma=gamma,
        theta_per_day=theta_annual / 365.0,
        vega_per_vol_point=vega,
        probability_itm=probability_itm,
        d1=d1,
        d2=d2,
    )


def expected_move(spot: float, annual_iv: float, time_years: float) -> float:
    return spot * annual_iv * math.sqrt(max(time_years, 0.0))


def probability_touch_approx(probability_itm: float) -> float:
    """Common rough approximation; not an exact first-passage calculation."""
    return min(1.0, max(0.0, 2.0 * probability_itm))


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No price history was returned.")

    out = df.copy()
    close = out["Close"].astype(float)

    out["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    out["SMA_50"] = close.rolling(50).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std(ddof=0)
    out["BB_MID"] = rolling_mean
    out["BB_UPPER"] = rolling_mean + 2 * rolling_std
    out["BB_LOWER"] = rolling_mean - 2 * rolling_std

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (out["High"] - out["Low"]).abs(),
            (out["High"] - previous_close).abs(),
            (out["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR_14"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    out["VOL_AVG_20"] = out["Volume"].rolling(20).mean()
    out["REL_VOLUME"] = out["Volume"] / out["VOL_AVG_20"].replace(0, np.nan)

    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3.0
    dates = pd.Series(out.index.date, index=out.index)
    cumulative_tpv = (typical_price * out["Volume"]).groupby(dates).cumsum()
    cumulative_volume = out["Volume"].groupby(dates).cumsum()
    out["VWAP"] = cumulative_tpv / cumulative_volume.replace(0, np.nan)

    out["RETURN_1"] = close.pct_change()
    out["REALIZED_VOL_20"] = out["RETURN_1"].rolling(20).std() * math.sqrt(252)

    return out


def technical_score(latest: pd.Series, previous: pd.Series) -> dict[str, float | str]:
    """Transparent directional score from -100 (bearish) to +100 (bullish)."""
    points = 0.0
    reasons: list[str] = []
    close = float(latest["Close"])

    comparisons = [
        ("EMA_9", 18, "above 9 EMA", "below 9 EMA"),
        ("VWAP", 18, "above VWAP", "below VWAP"),
        ("SMA_50", 12, "above 50 SMA", "below 50 SMA"),
    ]
    for column, weight, bull_text, bear_text in comparisons:
        value = latest.get(column)
        if pd.notna(value):
            if close > float(value):
                points += weight
                reasons.append(bull_text)
            else:
                points -= weight
                reasons.append(bear_text)

    macd = latest.get("MACD")
    signal = latest.get("MACD_SIGNAL")
    prev_hist = previous.get("MACD_HIST")
    hist = latest.get("MACD_HIST")
    if pd.notna(macd) and pd.notna(signal):
        if float(macd) > float(signal):
            points += 16
            reasons.append("MACD bullish")
        else:
            points -= 16
            reasons.append("MACD bearish")
    if pd.notna(hist) and pd.notna(prev_hist):
        if float(hist) > float(prev_hist):
            points += 6
            reasons.append("MACD momentum improving")
        else:
            points -= 6
            reasons.append("MACD momentum weakening")

    rsi = latest.get("RSI_14")
    if pd.notna(rsi):
        rsi = float(rsi)
        if 55 <= rsi <= 70:
            points += 12
            reasons.append("RSI supports bullish momentum")
        elif 30 <= rsi <= 45:
            points -= 12
            reasons.append("RSI supports bearish momentum")
        elif rsi > 75:
            points -= 4
            reasons.append("RSI potentially overbought")
        elif rsi < 25:
            points += 4
            reasons.append("RSI potentially oversold")

    rel_volume = latest.get("REL_VOLUME")
    candle_direction = np.sign(float(latest["Close"]) - float(latest["Open"]))
    if pd.notna(rel_volume) and float(rel_volume) >= 1.25:
        points += 8 * candle_direction
        reasons.append("high relative volume confirms latest candle")

    upper = latest.get("BB_UPPER")
    lower = latest.get("BB_LOWER")
    if pd.notna(upper) and close > float(upper):
        points += 5
        reasons.append("above upper Bollinger Band")
    elif pd.notna(lower) and close < float(lower):
        points -= 5
        reasons.append("below lower Bollinger Band")

    score = float(np.clip(points, -100, 100))
    direction = "bullish" if score >= 20 else "bearish" if score <= -20 else "neutral"
    return {"score": score, "direction": direction, "reasons": "; ".join(reasons)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return either (Price, Ticker) or (Ticker, Price)
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)
    return df


def download_price_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker = yf.Ticker(config.ticker)

    daily = ticker.history(period=config.daily_period, interval="1d", auto_adjust=False)
    intraday = ticker.history(
        period=config.intraday_period,
        interval=config.intraday_interval,
        auto_adjust=False,
        prepost=False,
    )

    daily = flatten_yfinance_columns(daily, config.ticker)
    intraday = flatten_yfinance_columns(intraday, config.ticker)

    if daily.empty:
        raise RuntimeError("Daily data download returned no rows.")
    if intraday.empty:
        raise RuntimeError("Intraday data download returned no rows.")

    return calculate_indicators(daily), calculate_indicators(intraday)


def choose_expiration(ticker: yf.Ticker, requested: str | None, min_dte: int, max_dte: int) -> str:
    expirations = list(ticker.options)
    if not expirations:
        raise RuntimeError("No option expiration dates were returned.")

    if requested:
        if requested not in expirations:
            raise ValueError(
                f"Expiration {requested} was not found. Available examples: {expirations[:8]}"
            )
        return requested

    now = datetime.now(timezone.utc).date()
    valid = []
    for expiration in expirations:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        dte = (exp_date - now).days
        if min_dte <= dte <= max_dte:
            valid.append((dte, expiration))

    if not valid:
        # Fall back to nearest nonexpired expiration.
        valid = [
            ((datetime.strptime(e, "%Y-%m-%d").date() - now).days, e)
            for e in expirations
            if (datetime.strptime(e, "%Y-%m-%d").date() - now).days >= 0
        ]
    if not valid:
        raise RuntimeError("No nonexpired option expiration was found.")

    return sorted(valid)[0][1]


# ---------------------------------------------------------------------------
# Options ranking
# ---------------------------------------------------------------------------

def safe_float(value: object, default: float = 0.0) -> float:
    """Convert Yahoo/pandas values safely, including None and NaN."""
    try:
        if value is None or pd.isna(value):
            return default
        converted = float(value)
        return converted if math.isfinite(converted) else default
    except (TypeError, ValueError, OverflowError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    """Convert Yahoo/pandas values safely, including float NaN."""
    try:
        if value is None or pd.isna(value):
            return default
        converted = float(value)
        if not math.isfinite(converted):
            return default
        return int(converted)
    except (TypeError, ValueError, OverflowError):
        return default


def safe_mid(row: pd.Series) -> float:
    bid = safe_float(row.get("bid"), 0.0)
    ask = safe_float(row.get("ask"), 0.0)
    last = safe_float(row.get("lastPrice"), 0.0)

    if bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    return last


def scenario_option_value(
    spot_scenario: float,
    strike: float,
    remaining_time_years: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: OptionType,
) -> float:
    return black_scholes_merton(
        spot=spot_scenario,
        strike=strike,
        time_years=max(remaining_time_years, 1 / (365 * 24 * 60)),
        rate=rate,
        volatility=max(volatility, 0.01),
        dividend_yield=dividend_yield,
        option_type=option_type,
    ).value


def build_option_table(
    config: Config,
    spot: float,
    expiration: str,
    technical: dict[str, float | str],
) -> pd.DataFrame:
    ticker = yf.Ticker(config.ticker)
    chain = ticker.option_chain(expiration)
    expiration_date = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    # Approximate expiration at 4 p.m. ET = 20:00 UTC during daylight time.
    expiration_time = expiration_date.replace(hour=20)
    seconds = max((expiration_time - now).total_seconds(), 60.0)
    t_years = seconds / (365.0 * 24 * 3600)
    dte = seconds / (24 * 3600)

    frames: list[pd.DataFrame] = []
    for option_type, raw in (("call", chain.calls), ("put", chain.puts)):
        if raw.empty:
            continue
        table = raw.copy()
        table["option_type"] = option_type
        frames.append(table)

    if not frames:
        raise RuntimeError("Option chain returned no contracts.")

    options = pd.concat(frames, ignore_index=True)
    records: list[dict[str, float | str | int]] = []

    tech_score = float(technical["score"])
    directional_alignment = {
        "call": max(0.0, (tech_score + 100.0) / 200.0),
        "put": max(0.0, (100.0 - tech_score) / 200.0),
    }

    for _, row in options.iterrows():
        option_type: OptionType = str(row["option_type"])  # type: ignore[assignment]
        strike = float(row["strike"])
        iv = safe_float(row.get("impliedVolatility"), float("nan"))
        premium = safe_mid(row)
        bid = safe_float(row.get("bid"), 0.0)
        ask = safe_float(row.get("ask"), 0.0)
        volume = safe_int(row.get("volume"), 0)
        open_interest = safe_int(row.get("openInterest"), 0)

        if not np.isfinite(iv) or iv <= 0 or premium <= 0:
            continue

        bs = black_scholes_merton(
            spot, strike, t_years, config.risk_free_rate, iv,
            config.dividend_yield, option_type
        )
        em = expected_move(spot, iv, t_years)
        direction = 1.0 if option_type == "call" else -1.0
        scenario_spot = max(0.01, spot + direction * config.scenario_move_sigma * em)

        # Estimate value after one trading day, or at expiration if sooner.
        remaining_after_day = max(t_years - 1 / 365.0, 1 / (365 * 24 * 60))
        scenario_value = scenario_option_value(
            scenario_spot, strike, remaining_after_day, config.risk_free_rate,
            iv, config.dividend_yield, option_type
        )
        scenario_return = scenario_value / premium - 1.0

        spread = max(ask - bid, 0.0)
        spread_pct = spread / premium if premium > 0 else np.inf
        liquidity_score = np.clip(
            0.50 * min(open_interest / max(config.min_open_interest, 1), 1.0)
            + 0.25 * min(volume / max(config.min_volume, 1), 1.0)
            + 0.25 * max(0.0, 1.0 - spread_pct / max(config.max_spread_pct, 0.01)),
            0.0,
            1.0,
        )
        probability_score = np.clip(bs.probability_itm, 0.0, 1.0)
        payoff_score = np.clip((scenario_return + 1.0) / 3.0, 0.0, 1.0)
        theta_burden = min(abs(bs.theta_per_day) / premium, 1.0)
        affordability = 1.0 if premium * 100 <= config.account_size * config.max_risk_pct else 0.25

        # Breakeven at expiration and max loss, assuming a long option position
        # held to expiration (the max loss for a buyer is always the premium paid).
        breakeven_price = strike + premium if option_type == "call" else strike - premium
        max_loss_dollars = premium * 100

        # Transparent heuristic—not a forecast.
        composite = 100 * (
            0.25 * directional_alignment[option_type]
            + 0.20 * probability_score
            + 0.20 * liquidity_score
            + 0.20 * payoff_score
            + 0.10 * (1.0 - theta_burden)
            + 0.05 * affordability
        )

        records.append(
            {
                "contractSymbol": row.get("contractSymbol", ""),
                "expiration": expiration,
                "DTE": dte,
                "option_type": option_type,
                "strike": strike,
                "spot": spot,
                "bid": bid,
                "ask": ask,
                "mid": premium,
                "contract_cost": premium * 100,
                "volume": volume,
                "open_interest": open_interest,
                "spread_pct": spread_pct,
                "IV": iv,
                "BS_value": bs.value,
                "mispricing_vs_mid": bs.value / premium - 1.0,
                "delta": bs.delta,
                "gamma": bs.gamma,
                "theta_per_day": bs.theta_per_day,
                "theta_pct_of_premium": abs(bs.theta_per_day) / premium,
                "vega_per_vol_point": bs.vega_per_vol_point,
                "probability_ITM": bs.probability_itm,
                "probability_touch_approx": probability_touch_approx(bs.probability_itm),
                "expected_move": em,
                "scenario_spot_1sigma": scenario_spot,
                "scenario_value_after_1day": scenario_value,
                "scenario_return_after_1day": scenario_return,
                "technical_score": tech_score,
                "technical_direction": technical["direction"],
                "liquidity_score": liquidity_score,
                "decision_score": composite,
                "target_100pct_scenario": scenario_return >= config.target_return,
                "within_risk_budget": premium * 100 <= config.account_size * config.max_risk_pct,
                "breakeven_price": breakeven_price,
                "breakeven_move_pct": (breakeven_price / spot - 1.0),
                "max_loss_dollars": max_loss_dollars,
            }
        )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise RuntimeError("No valid contracts remained after pricing checks.")

    # Basic practical filters; preserve all contracts in a separate export.
    result["passes_liquidity_filter"] = (
        (result["open_interest"] >= config.min_open_interest)
        & (result["volume"] >= config.min_volume)
        & (result["spread_pct"] <= config.max_spread_pct)
    )
    result["distance_from_spot_pct"] = (result["strike"] / spot - 1.0).abs()
    result = result.sort_values(
        ["passes_liquidity_filter", "decision_score", "distance_from_spot_pct"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# IV rank, skew, and scoring-accuracy history
#
# yfinance does not expose historical implied volatility, so a textbook
# 52-week IV rank isn't available from a cold start. Instead, this script
# keeps its own running log of ATM IV and technical score on every scan.
# IV rank and score-accuracy therefore start out based on whatever history
# exists (labeled honestly as such) and become more meaningful the longer
# you run scans, since the log accumulates over time.
# ---------------------------------------------------------------------------

def get_atm_iv(options: pd.DataFrame, spot: float) -> float | None:
    """Average IV of the call and put contracts closest to the money."""
    if options.empty:
        return None
    nearest_idx = (options["strike"] - spot).abs().sort_values().index
    nearest = options.loc[nearest_idx[:4]]  # a few closest strikes, both types
    ivs = nearest["IV"].dropna()
    return float(ivs.mean()) if not ivs.empty else None


def compute_skew(options: pd.DataFrame, spot: float, target_pct: float) -> dict[str, float]:
    """Compare IV of an OTM put vs. an OTM call roughly equidistant from spot.

    A positive skew (put IV > call IV) is the normal equity/index pattern and
    reflects richer pricing for downside protection. A flattening or inverted
    skew can signal shifting sentiment, independent of the technical score.
    """
    calls = options[options["option_type"] == "call"]
    puts = options[options["option_type"] == "put"]
    if calls.empty or puts.empty:
        return {}

    target_call_strike = spot * (1 + target_pct)
    target_put_strike = spot * (1 - target_pct)
    call_row = calls.iloc[(calls["strike"] - target_call_strike).abs().to_numpy().argsort()[:1]]
    put_row = puts.iloc[(puts["strike"] - target_put_strike).abs().to_numpy().argsort()[:1]]
    if call_row.empty or put_row.empty:
        return {}

    call_iv = float(call_row["IV"].iloc[0])
    put_iv = float(put_row["IV"].iloc[0])
    return {
        "call_strike": float(call_row["strike"].iloc[0]),
        "put_strike": float(put_row["strike"].iloc[0]),
        "call_iv": call_iv,
        "put_iv": put_iv,
        "skew": put_iv - call_iv,
    }


def load_history_log(history_dir: Path, ticker: str) -> pd.DataFrame:
    path = history_dir / f"{ticker}_scan_history.csv"
    columns = ["date", "ticker", "spot", "atm_iv", "technical_score"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame(columns=columns)


def append_history_log(
    history_dir: Path,
    ticker: str,
    spot: float,
    atm_iv: float | None,
    technical_score: float,
) -> pd.DataFrame:
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{ticker}_scan_history.csv"
    existing = load_history_log(history_dir, ticker)
    new_row = pd.DataFrame(
        [{
            "date": pd.Timestamp.now(tz="UTC").tz_localize(None),
            "ticker": ticker,
            "spot": spot,
            "atm_iv": atm_iv,
            "technical_score": technical_score,
        }]
    )
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined


def compute_iv_rank(history: pd.DataFrame, current_iv: float | None) -> dict[str, float] | None:
    """Percentile rank of the current ATM IV against the logged history.

    With a short history this is a rank against a small sample, not a true
    52-week rank — the sample size (n) is always included so that's clear.
    """
    if current_iv is None:
        return None
    ivs = history["atm_iv"].dropna()
    if len(ivs) < 3:
        return None
    rank = float((ivs <= current_iv).mean() * 100)
    return {"iv_rank": rank, "n": int(len(ivs)), "iv_min": float(ivs.min()), "iv_max": float(ivs.max())}


def evaluate_score_history(
    history: pd.DataFrame,
    daily_prices: pd.DataFrame,
    forward_days: tuple[int, ...] = (1, 3, 5),
    min_samples: int = 10,
) -> dict[int, dict[str, float]]:
    """Backtest the technical score against what QQQ actually did afterward.

    For each past scan, looks forward N trading days in the daily price
    history and checks: did the sign of the score match the sign of the
    subsequent return, and how correlated were score and return overall.
    This does NOT validate the option-pricing math (Black-Scholes is a
    standard formula) — it validates whether the heuristic technical score
    has actually been predictive, which is the part worth checking.
    """
    results: dict[int, dict[str, float]] = {}
    if history.empty or len(history) < min_samples:
        return results

    close = daily_prices["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    for horizon in forward_days:
        pairs: list[tuple[float, float]] = []
        for _, row in history.iterrows():
            entry_date = pd.Timestamp(row["date"]).tz_localize(None)
            prior = close[close.index <= entry_date]
            future = close[close.index > entry_date]
            if prior.empty or len(future) < horizon:
                continue
            entry_price = float(prior.iloc[-1])
            future_price = float(future.iloc[horizon - 1])
            forward_return = future_price / entry_price - 1.0
            pairs.append((float(row["technical_score"]), forward_return))

        if len(pairs) < min_samples:
            continue
        scores, returns = zip(*pairs)
        scores_arr = np.array(scores)
        returns_arr = np.array(returns)
        correlation = float(np.corrcoef(scores_arr, returns_arr)[0, 1]) if np.std(scores_arr) > 0 else 0.0
        nonzero = scores_arr != 0
        hit_rate = (
            float(np.mean((scores_arr[nonzero] > 0) == (returns_arr[nonzero] > 0)))
            if nonzero.any() else 0.0
        )
        results[horizon] = {"n": len(pairs), "correlation": correlation, "hit_rate": hit_rate}

    return results


# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------

def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    """Send a text message via a Telegram bot.

    Logs a warning on failure instead of crashing the run, since a failed
    notification shouldn't take down the whole scan.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"WARNING: Telegram message failed to send: {exc}", file=sys.stderr)


def send_telegram_photo(token: str, chat_id: str, photo_path: Path, caption: str = "") -> None:
    """Send a photo (e.g. a chart) via a Telegram bot."""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(photo_path, "rb") as photo_file:
            response = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": photo_file},
                timeout=30,
            )
        response.raise_for_status()
    except (requests.RequestException, OSError) as exc:
        print(f"WARNING: Telegram photo failed to send: {exc}", file=sys.stderr)


def format_top_contracts_for_telegram(
    options: pd.DataFrame,
    technical: dict[str, float | str],
    spot: float,
    ticker: str,
    iv_rank_info: dict[str, float] | None = None,
    skew_info: dict[str, float] | None = None,
    score_accuracy: dict[int, dict[str, float]] | None = None,
    top_n: int = 5,
) -> str:
    lines = [
        f"*{ticker} Options Scan*",
        f"Spot: ${spot:,.2f} | Technical: {technical['score']:.0f}/100 ({technical['direction']})",
    ]

    if iv_rank_info:
        lines.append(
            f"ATM IV rank: {iv_rank_info['iv_rank']:.0f}th pct (n={iv_rank_info['n']} scans)"
        )
    if skew_info:
        skew_label = "put-skewed" if skew_info["skew"] > 0 else "call-skewed"
        lines.append(
            f"Skew: {skew_info['skew']*100:+.1f} vol pts ({skew_label})"
        )
    if score_accuracy:
        parts = [
            f"{h}d hit-rate {v['hit_rate']:.0%} (n={v['n']})"
            for h, v in sorted(score_accuracy.items())
        ]
        lines.append("Score accuracy: " + ", ".join(parts))

    lines.append("")
    for row in options.head(top_n).itertuples():
        liquidity_flag = "✅" if row.passes_liquidity_filter else "⚠️"
        lines.append(
            f"{liquidity_flag} {row.option_type.upper()} {row.strike:g}  exp {row.expiration}\n"
            f"   Mid ${row.mid:.2f}  |  Score {row.decision_score:.0f}/100\n"
            f"   P(ITM) {row.probability_ITM:.0%}  |  DTE {row.DTE:.1f}  |  Θ/day {row.theta_pct_of_premium:.1%} of premium\n"
            f"   Breakeven ${row.breakeven_price:.2f} ({row.breakeven_move_pct:+.1%} move)  |  Max loss ${row.max_loss_dollars:.0f}"
        )
    lines.append("")
    lines.append("_Heuristic score, not a trade signal. Not financial advice._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reports and plots
# ---------------------------------------------------------------------------

def save_price_chart(df: pd.DataFrame, output_path: Path, title: str) -> None:
    view = df.tail(120).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(view.index, view["Close"], label="Close")
    ax.plot(view.index, view["EMA_9"], label="9 EMA")
    ax.plot(view.index, view["VWAP"], label="VWAP")
    if view["SMA_50"].notna().any():
        ax.plot(view.index, view["SMA_50"], label="50 SMA")
    ax.fill_between(
        view.index,
        view["BB_LOWER"].to_numpy(dtype=float),
        view["BB_UPPER"].to_numpy(dtype=float),
        alpha=0.12,
        label="Bollinger Bands",
    )
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_score_chart(options: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    view = options.head(top_n).sort_values("decision_score")
    labels = [
        f"{row.option_type.upper()} {row.strike:g} ({row.DTE:.1f} DTE)"
        for row in view.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, view["decision_score"])
    ax.set_title("Highest Heuristic Decision Scores")
    ax.set_xlabel("Score (0–100)")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(
    config: Config,
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    technical: dict[str, float | str],
    expiration: str,
    options: pd.DataFrame,
    iv_rank_info: dict[str, float] | None = None,
    skew_info: dict[str, float] | None = None,
    score_accuracy: dict[int, dict[str, float]] | None = None,
) -> None:
    latest = intraday.iloc[-1]
    spot = float(latest["Close"])
    daily_atr = float(daily["ATR_14"].dropna().iloc[-1])
    rv = float(daily["REALIZED_VOL_20"].dropna().iloc[-1])

    print("\n" + "=" * 78)
    print(f"{config.ticker} OPTIONS DECISION MODEL")
    print("=" * 78)
    print(f"Price timestamp:       {intraday.index[-1]}")
    print(f"Underlying price:      ${spot:,.2f}")
    print(f"Daily ATR(14):         ${daily_atr:,.2f}")
    print(f"20-day realized vol:   {rv:.2%}")
    print(f"Technical score:       {technical['score']:.1f}/100 ({technical['direction']})")
    print(f"Technical evidence:    {technical['reasons']}")
    print(f"Selected expiration:   {expiration}")

    if iv_rank_info:
        print(
            f"ATM IV rank:           {iv_rank_info['iv_rank']:.0f}th percentile "
            f"(n={iv_rank_info['n']} logged scans, range {iv_rank_info['iv_min']:.1%}-{iv_rank_info['iv_max']:.1%})"
        )
    else:
        print("ATM IV rank:           not enough logged history yet (needs 3+ scans)")

    if skew_info:
        skew_label = "put-skewed (normal)" if skew_info["skew"] > 0 else "call-skewed"
        print(
            f"Put/call skew:         {skew_info['skew']*100:+.1f} vol pts "
            f"(put {skew_info['put_strike']:g} IV {skew_info['put_iv']:.1%} vs "
            f"call {skew_info['call_strike']:g} IV {skew_info['call_iv']:.1%}) — {skew_label}"
        )

    if score_accuracy:
        print("Technical score backtest (vs. actual forward QQQ return):")
        for horizon, stats in sorted(score_accuracy.items()):
            print(
                f"  {horizon}-day: hit-rate {stats['hit_rate']:.0%}, "
                f"correlation {stats['correlation']:+.2f}, n={stats['n']}"
            )
    else:
        print("Technical score backtest: not enough logged history yet (needs 10+ scans)")

    print("\nTop contracts by transparent heuristic score:")
    columns = [
        "option_type", "strike", "DTE", "mid", "contract_cost", "IV",
        "probability_ITM", "theta_pct_of_premium",
        "scenario_return_after_1day", "breakeven_price", "max_loss_dollars",
        "decision_score", "passes_liquidity_filter", "within_risk_budget",
    ]
    display = options[columns].head(12).copy()
    for col in ["IV", "probability_ITM", "theta_pct_of_premium", "scenario_return_after_1day"]:
        display[col] = display[col].map(lambda x: f"{x:.1%}")
    display["mid"] = display["mid"].map(lambda x: f"${x:.2f}")
    display["contract_cost"] = display["contract_cost"].map(lambda x: f"${x:.0f}")
    display["breakeven_price"] = display["breakeven_price"].map(lambda x: f"${x:.2f}")
    display["max_loss_dollars"] = display["max_loss_dollars"].map(lambda x: f"${x:.0f}")
    display["decision_score"] = display["decision_score"].map(lambda x: f"{x:.1f}")
    print(display.to_string(index=False))
    print("\nCAUTION: The score is a rule-based comparison tool, not a buy/sell signal.")
    print("A 100% target generally requires a large, fast move and can also produce a 100% loss.")
    print("IV rank and score backtest are based on this script's own logged history, not a full market dataset.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze QQQ price action and rank option contracts."
    )
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--expiration", default=None, help="YYYY-MM-DD; defaults to nearest expiration in DTE range")
    parser.add_argument("--min-dte", type=int, default=1)
    parser.add_argument("--max-dte", type=int, default=10)
    parser.add_argument("--risk-free-rate", type=float, default=0.045)
    parser.add_argument("--dividend-yield", type=float, default=0.005)
    parser.add_argument("--account-size", type=float, default=300.0)
    parser.add_argument("--max-risk-pct", type=float, default=0.10)
    parser.add_argument("--min-open-interest", type=int, default=100)
    parser.add_argument("--min-volume", type=int, default=20)
    parser.add_argument("--max-spread-pct", type=float, default=0.20)
    parser.add_argument("--output-dir", default="qqq_model_output")
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Skip sending results to Telegram even if credentials are set.",
    )
    parser.add_argument(
        "--telegram-top-n",
        type=int,
        default=5,
        help="Number of top contracts to include in the Telegram message.",
    )
    parser.add_argument(
        "--telegram-send-chart",
        action="store_true",
        help="Also send the option score chart as a photo to Telegram.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config(
        ticker=args.ticker.upper(),
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
        account_size=args.account_size,
        max_risk_pct=args.max_risk_pct,
        min_open_interest=args.min_open_interest,
        min_volume=args.min_volume,
        max_spread_pct=args.max_spread_pct,
        output_dir=Path(args.output_dir),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Telegram credentials come from environment variables (e.g. GitHub Actions
    # secrets), never from command-line args or hardcoded values, to avoid
    # leaking them into shell history or version control.
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    telegram_enabled = bool(telegram_token and telegram_chat_id) and not args.no_telegram

    try:
        daily, intraday = download_price_data(config)
        technical = technical_score(intraday.iloc[-1], intraday.iloc[-2])
        spot = float(intraday["Close"].iloc[-1])

        ticker = yf.Ticker(config.ticker)
        expiration = choose_expiration(ticker, args.expiration, args.min_dte, args.max_dte)
        options = build_option_table(config, spot, expiration, technical)

        # IV rank, skew, and score-accuracy backtest, all built from this
        # script's own accumulated history rather than a paid data source.
        atm_iv = get_atm_iv(options, spot)
        skew_info = compute_skew(options, spot, config.skew_target_pct)
        history = append_history_log(config.history_dir, config.ticker, spot, atm_iv, technical["score"])
        iv_rank_info = compute_iv_rank(history, atm_iv)
        score_accuracy = evaluate_score_history(history, daily)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        daily.to_csv(config.output_dir / f"{config.ticker}_daily_{timestamp}.csv")
        intraday.to_csv(config.output_dir / f"{config.ticker}_intraday_{timestamp}.csv")
        options.to_csv(config.output_dir / f"{config.ticker}_option_rankings_{timestamp}.csv", index=False)

        intraday_chart_path = config.output_dir / f"{config.ticker}_intraday_chart_{timestamp}.png"
        score_chart_path = config.output_dir / f"{config.ticker}_option_scores_{timestamp}.png"

        save_price_chart(
            intraday,
            intraday_chart_path,
            f"{config.ticker} intraday price and indicators",
        )
        save_score_chart(options, score_chart_path)
        print_summary(
            config, daily, intraday, technical, expiration, options,
            iv_rank_info=iv_rank_info, skew_info=skew_info, score_accuracy=score_accuracy,
        )
        print(f"\nFiles saved in: {config.output_dir.resolve()}")

        if telegram_enabled:
            message = format_top_contracts_for_telegram(
                options, technical, spot, config.ticker,
                iv_rank_info=iv_rank_info, skew_info=skew_info, score_accuracy=score_accuracy,
                top_n=args.telegram_top_n,
            )
            send_telegram_message(telegram_token, telegram_chat_id, message)
            if args.telegram_send_chart:
                send_telegram_photo(
                    telegram_token, telegram_chat_id, score_chart_path,
                    caption=f"{config.ticker} top contract scores",
                )
            print("Telegram notification sent.")
        elif not args.no_telegram:
            print(
                "Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
                "env vars to enable). Skipping notification.",
                file=sys.stderr,
            )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Check your internet connection, ticker, expiration, and whether "
            "Yahoo Finance is returning an option chain.",
            file=sys.stderr,
        )
        if telegram_enabled:
            send_telegram_message(
                telegram_token, telegram_chat_id,
                f"*{config.ticker} scan failed*\n`{exc}`",
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())