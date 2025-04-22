
import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt

st.set_page_config(page_title="시장 심리 대시보드", layout="wide")
st.title("📈 시장 심리 대시보드: Buffett Indicator & Fear & Greed Index")
st.markdown("미국 시장의 심리적 상태를 두 가지 핵심 지표로 한눈에 파악하세요.")

# DuckDB 연결
con = duckdb.connect(database='buffett_indicator.duckdb', read_only=True)

# Buffett Indicator 데이터 로드
st.subheader("📊 Buffett Indicator & Wilshire 5000 Market Cap")
buffett_df = con.execute("SELECT * FROM buffett_indicator_data").fetchdf()

fig1, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(buffett_df['Date'], buffett_df['Buffett_Indicator'], color='blue', linewidth=1, label='Buffett Indicator (%)')
ax1.fill_between(buffett_df['Date'], 100, 120, color='orange', alpha=0.1)
ax1.fill_between(buffett_df['Date'], 120, buffett_df['Buffett_Indicator'].max(), color='red', alpha=0.1)
ax1.axhline(y=100, color='gray', linestyle='--', linewidth=1)
ax1.axhline(y=120, color='red', linestyle='--', linewidth=1)
ax1.set_xlabel('Date')
ax1.set_ylabel('Buffett Indicator (%)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(buffett_df['Date'], buffett_df['Market_Cap'], color='purple', linewidth=1, alpha=0.6, label='Wilshire 5000 Market Cap')
ax2.set_ylabel('Wilshire 5000 Market Cap', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

fig1.tight_layout()
st.pyplot(fig1)

# Fear & Greed Index + S&P500 데이터 로드
st.subheader("📊 Fear & Greed Index & S&P 500 Index")
fng_df = con.execute("SELECT * FROM fear_greed_index_data").fetchdf()
sp500_df = con.execute("SELECT * FROM sp500_data").fetchdf()

# 전처리
fng_df['Date'] = pd.to_datetime(fng_df['Date']).dt.date
sp500_df['Date'] = pd.to_datetime(sp500_df['Date']).dt.date

# 병합
available_dates = fng_df['Date']
filtered_sp500_df = sp500_df[sp500_df['Date'].isin(available_dates)]
merged_sp500 = pd.merge(fng_df, filtered_sp500_df, on='Date', how='inner')

fig2, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(merged_sp500['Date'], merged_sp500['FearGreedValue'], color='green', linewidth=1, alpha=0.6, label='Fear & Greed Index')
ax1.set_xlabel('Date')
ax1.set_ylabel('Fear & Greed Index', color='green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.set_ylim(0, 100)

# 공포/탐욕 구간 강조
ax1.axhspan(0, 20, color='red', alpha=0.1)
ax1.axhspan(20, 40, color='orange', alpha=0.1)
ax1.axhspan(40, 60, color='lightgreen', alpha=0.1)
ax1.axhspan(60, 100, color='green', alpha=0.1)

ax2 = ax1.twinx()
ax2.plot(merged_sp500['Date'], merged_sp500['SP500_Close'], color='orange', linewidth=1, alpha=0.6, label='S&P 500 Close')
ax2.set_ylabel('S&P 500 Index', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

fig2.tight_layout()
st.pyplot(fig2)

st.success("✅ 대시보드가 성공적으로 로드되었습니다!")
