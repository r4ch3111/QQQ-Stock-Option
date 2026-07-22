import streamlit as st
import subprocess
import sys
from pathlib import Path

st.set_page_config(
page_title="QQQ Options Model",
page_icon="📈",
layout="wide",
)

st.title("QQQ Options Decision Model")
st.caption("Educational analysis tool — not a guaranteed trading signal.")

ticker = st.text_input("Ticker", value="QQQ").upper()

col1, col2 = st.columns(2)

with col1:
minimum_dte = st.number_input(
"Minimum days to expiration",
min_value=0,
max_value=60,
value=1,
)

with col2:
maximum_dte = st.number_input(
"Maximum days to expiration",
min_value=1,
max_value=90,
value=10,
)

account_size = st.number_input(
"Account size",
min_value=1.0,
value=300.0,
step=50.0,
)

maximum_risk = st.slider(
"Maximum risk per trade",
min_value=0.01,
max_value=1.00,
value=0.10,
format="%.0f%%",
)

if st.button("Run Analysis", type="primary"):
if minimum_dte > maximum_dte:
st.error("Minimum DTE cannot be greater than maximum DTE.")
else:
with st.spinner("Downloading QQQ and option-chain data..."):
command = [
sys.executable,
"qqq_options_model.py",
"--ticker",
ticker,
"--min-dte",
str(minimum_dte),
"--max-dte",
str(maximum_dte),
"--account-size",
str(account_size),
"--max-risk-pct",
str(maximum_risk),
"--output-dir",
"qqq_model_output",
]

result = subprocess.run(
command,
capture_output=True,
text=True,
)

if result.returncode != 0:
st.error("The analysis could not be completed.")
st.code(result.stderr)
else:
st.success("Analysis completed.")
st.code(result.stdout)

output_folder = Path("qqq_model_output")

csv_files = sorted(
output_folder.glob("*option_rankings*.csv"),
key=lambda file: file.stat().st_mtime,
reverse=True,
)

chart_files = sorted(
output_folder.glob("*intraday_chart*.png"),
key=lambda file: file.stat().st_mtime,
reverse=True,
)

if chart_files:
st.subheader("QQQ Chart")
st.image(str(chart_files[0]), use_container_width=True)

if csv_files:
import pandas as pd

rankings = pd.read_csv(csv_files[0])

st.subheader("Highest-Ranked Options")

preferred_columns = [
"option_type",
"strike",
"DTE",
"mid",
"contract_cost",
"IV",
"delta",
"theta_per_day",
"probability_ITM",
"scenario_return_after_1day",
"decision_score",
"passes_liquidity_filter",
"within_risk_budget",
]

available_columns = [
column
for column in preferred_columns
if column in rankings.columns
]

st.dataframe(
rankings[available_columns].head(25),
use_container_width=True,
hide_index=True,
)

st.download_button(
"Download Full Results",
data=rankings.to_csv(index=False),
file_name=f"{ticker}_option_rankings.csv",
mime="text/csv",
)
else:
st.warning("No option-ranking file was created.")