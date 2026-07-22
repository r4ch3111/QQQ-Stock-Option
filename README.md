# QQQ Options Decision Model

This educational Python tool downloads QQQ price and options data, calculates
technical indicators and Black-Scholes-Merton estimates, then ranks available
contracts using a transparent heuristic score.

## What it calculates

- 9 EMA, VWAP and 50 SMA
- RSI(14), MACD, Bollinger Bands and ATR(14)
- Relative volume and 20-day realized volatility
- Black-Scholes theoretical value
- Delta, gamma, theta and vega
- Expected move
- Risk-neutral probability of finishing in the money
- Approximate probability of touching the strike
- Bid/ask spread and liquidity score
- One-standard-deviation scenario return
- A 0–100 comparison score

The score is **not** a prediction or automatic trading recommendation.

## Installation

Open Terminal or PowerShell in this folder:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run it

Nearest QQQ expiration between 1 and 10 days:

```bash
python qqq_options_model.py
```

Choose an expiration:

```bash
python qqq_options_model.py --expiration 2026-07-24
```

Use a different ticker:

```bash
python qqq_options_model.py --ticker SPY
```

Adjust account and risk limits:

```bash
python qqq_options_model.py --account-size 300 --max-risk-pct 0.10
```

This treats 10% of a $300 account, or $30, as the normal maximum premium risk.
That filter is intentionally conservative. Change it consciously rather than
simply increasing it until a contract fits.

## Output

The script creates a folder named `qqq_model_output` containing:

- Daily price and indicator CSV
- Intraday 5-minute price and indicator CSV
- Full option-ranking CSV
- Intraday price chart
- Option-score chart

## How the score works

The score combines:

- 25% technical-direction alignment
- 20% risk-neutral probability ITM
- 20% liquidity
- 20% one-standard-deviation payoff
- 10% resistance to theta decay
- 5% affordability

This weighting is only a starting framework. Use the exported journal/results
to test whether the weighting actually improves your decisions.

## Important limitations

1. Black-Scholes is a pricing model, not a direction predictor.
2. QQQ options are American-style; the script uses Black-Scholes-Merton as an
   approximation.
3. Yahoo/yfinance data may be delayed, incomplete, stale or temporarily
   unavailable.
4. Implied volatility can change rapidly, especially around major events.
5. Bid/ask spreads can make actual fills worse than the calculated midpoint.
6. “Probability ITM” is risk-neutral model probability, not a guaranteed
   real-world probability.
7. The approximate probability of touching a strike is a rough shortcut.
8. A contract capable of gaining 100% can lose 100% of its premium.

## Suggested workflow

1. Run after the first 15–30 minutes of the regular session.
2. Confirm that data timestamps and spreads are current.
3. Examine only contracts passing the liquidity filter.
4. Compare 1–3 DTE with 5–10 DTE rather than choosing only the cheapest premium.
5. Record the score, entry, exit and actual result.
6. Review at least 50–100 trades before changing weights.

## Data and math references

- yfinance project documentation: https://ranaroussi.github.io/yfinance/
- SciPy normal distribution documentation:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
- Options Industry Council education: https://www.optionseducation.org/
- Cboe Options Institute: https://www.cboe.com/optionsinstitute/
