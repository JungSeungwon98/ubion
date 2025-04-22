
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import duckdb

st.title("📈 Buffett Indicator Dashboard")

# DuckDB 에서 데이터 불러오기
con = duckdb.connect(database='buffett_indicator.duckdb', read_only=True)
buffett_df = con.execute('SELECT * FROM buffett_indicator_data').fetchdf()

st.line_chart(buffett_df.set_index('Date')['Buffett_Indicator'])

st.markdown("""
- ✅ **100% 이하:** 시장이 저평가 상태일 수 있음
- 🟡 **100% ~ 120%:** 합리적 수준
- 🔥 **120% 이상:** 시장 과열 구간
""")

fig, ax = plt.subplots()
ax.plot(buffett_df['Date'], buffett_df['Buffett_Indicator'], label='Buffett Indicator (%)')
ax.axhline(y=100, color='gray', linestyle='--', label='Fair Value (100%)')
ax.axhline(y=120, color='red', linestyle='--', label='Overheated (120%)')
ax.set_xlabel('Date')
ax.set_ylabel('Indicator (%)')
ax.legend()
st.pyplot(fig)
