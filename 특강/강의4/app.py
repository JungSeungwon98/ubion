import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Streamlit 페이지 설정
st.set_page_config(page_title='기업 재무 분석 및 예측 대시보드', layout='wide')
st.title('📊 기업 재무 분석 및 예측 대시보드')
st.markdown('사업자번호 또는 기업명을 입력하여 기업을 검색하고 성장성 및 수익성, 그리고 예측 결과를 확인하세요.')

# 폰트 설정
font_dirs = ['/usr/share/fonts/truetype/nanum/']
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    fm.fontManager.addfont(font_file)

nanum_font = fm.FontProperties(fname=font_files[0]).get_name()
plt.rcParams['font.family'] = nanum_font
plt.rcParams['axes.unicode_minus'] = False

# ✅ DuckDB 연결 및 데이터 로딩
@st.cache_data
def load_data():
    con = duckdb.connect(database='company_data.duckdb', read_only=False)
    company_df = con.execute("SELECT * FROM company_data").df()
    prediction_df = con.execute("SELECT * FROM prediction_results").df()
    con.close()
    return company_df, prediction_df

company_data, prediction_data = load_data()

# ✅ 데이터가 존재하는 기업만 필터링
valid_biz_no = company_data.dropna(subset=['손익계산서_매출액', '손익계산서_영업이익']).사업자번호.unique()

company_data_filtered = company_data[company_data['사업자번호'].isin(valid_biz_no)]
prediction_data_filtered = prediction_data[prediction_data['사업자번호'].isin(valid_biz_no)]

# ✅ 사용자 입력: 사업자번호 또는 기업명 검색
search_input = st.text_input('🔍 사업자번호 또는 기업명을 입력하세요')

if search_input:
    filtered_data = company_data_filtered[
        company_data_filtered['사업자번호'].astype(str).str.contains(search_input) |
        company_data_filtered['기업명'].str.contains(search_input, na=False)
    ].drop_duplicates(subset=['사업자번호'])

    if not filtered_data.empty:
        st.write(f"🔎 {len(filtered_data)}개의 기업이 검색되었습니다.")
        selected_company_name = st.selectbox('✅ 분석할 기업명을 선택하세요:', filtered_data['기업명'].unique())

        selected_biz_no = filtered_data[filtered_data['기업명'] == selected_company_name]['사업자번호'].values[0]

        company_df = company_data_filtered[company_data_filtered['사업자번호'] == selected_biz_no].sort_values('기준연도')
        prediction_df = prediction_data_filtered[prediction_data_filtered['사업자번호'] == selected_biz_no].sort_values('기준연도')

        selected_industry = company_df['업종'].iloc[0] if not company_df.empty else '정보 없음'
        st.markdown(f"### 📌 선택된 기업: {selected_company_name} (사업자번호: {selected_biz_no})")
        st.markdown(f"#### 🔹 소속 업종: {selected_industry}")

        industry_df = company_data_filtered[company_data_filtered['업종'] == selected_industry].groupby('기준연도').mean(numeric_only=True).reset_index()

        # ✅ 차트
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('📈 매출 증가율 추이 (과거 + 예측)')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(company_df['기준연도'], company_df['매출증가율'], marker='o', label='과거 데이터', color='skyblue')
            ax.scatter(prediction_df['기준연도'], prediction_df['예측_매출증가율'], color='orange', label='예측', zorder=5)
            ax.set_xlabel('기준연도')
            ax.set_ylabel('매출 증가율 (%)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with col2:
            st.subheader('📈 영업이익률 추이 (과거 + 예측)')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(company_df['기준연도'], company_df['영업이익률'], marker='o', label='과거 데이터', color='salmon')
            ax.scatter(prediction_df['기준연도'], prediction_df['예측_영업이익률'], color='orange', label='예측', zorder=5)
            ax.set_xlabel('기준연도')
            ax.set_ylabel('영업이익률 (%)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # ✅ 매출액 & 영업이익 차트 추가
        col3, col4 = st.columns(2)

        with col3:
          st.subheader('💰 매출액 추이 (과거 + 예측)')
          fig, ax = plt.subplots(figsize=(6, 4))
          ax.plot(company_df['기준연도'], company_df['손익계산서_매출액'], marker='o', label='과거 매출액', color='green')

          # ✅ 예측 매출액 계산
          if not prediction_df.empty and '예측_매출증가율' in prediction_df.columns:
              future_years = prediction_df['기준연도'].values
              # 과거 마지막 매출액
              last_sales = company_df['손익계산서_매출액'].iloc[-1]
              # 누적 예측 매출액 계산
              predicted_sales = [last_sales]
              for growth_rate in prediction_df['예측_매출증가율']:
                  next_sales = predicted_sales[-1] * (1 + growth_rate / 100)
                  predicted_sales.append(next_sales)
              predicted_sales.pop(0)  # 첫 값 제거 (last_sales 중복)

              ax.scatter(future_years, predicted_sales, color='orange', label='예측 매출액', zorder=5)

          ax.set_xlabel('기준연도')
          ax.set_ylabel('매출액')
          ax.legend()
          ax.grid(True)
          st.pyplot(fig)

        with col4:
          st.subheader('💼 영업이익 추이 (과거 + 예측)')
          fig, ax = plt.subplots(figsize=(6, 4))
          ax.plot(company_df['기준연도'], company_df['손익계산서_영업이익'], marker='o', label='과거 영업이익', color='purple')

          # ✅ 예측 영업이익 계산
          if not prediction_df.empty and '예측_영업이익률' in prediction_df.columns:
              future_years = prediction_df['기준연도'].values
              # 과거 마지막 매출액 및 영업이익률
              last_sales = company_df['손익계산서_매출액'].iloc[-1]
              predicted_sales = [last_sales]
              last_profit_rate = company_df['영업이익률'].iloc[-1]

              predicted_profits = []
              for growth_rate, profit_rate in zip(prediction_df['예측_매출증가율'], prediction_df['예측_영업이익률']):
                  # 매출액 누적 예측
                  next_sales = predicted_sales[-1] * (1 + growth_rate / 100)
                  predicted_sales.append(next_sales)

                  # 영업이익 = 매출액 * 영업이익률
                  next_profit = next_sales * (profit_rate / 100)
                  predicted_profits.append(next_profit)

              predicted_sales.pop(0)  # 첫 번째 중복 제거
              ax.scatter(future_years, predicted_profits, color='orange', label='예측 영업이익', zorder=5)

          ax.set_xlabel('기준연도')
          ax.set_ylabel('영업이익')
          ax.legend()
          ax.grid(True)
          st.pyplot(fig)
        # ✅ 데이터 테이블 표시
        st.subheader('📋 기업 데이터 테이블 (과거 + 예측)')
        combined_df = pd.concat([company_df, prediction_df], sort=False)
        st.dataframe(combined_df)

        st.success('✅ 대시보드가 성공적으로 로드되었습니다!')
    else:
        st.warning("검색 결과가 없습니다. 다시 입력해주세요.")
else:
    st.info("사업자번호 또는 기업명을 입력해주세요.")