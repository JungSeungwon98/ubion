### app.py
import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Streamlit 페이지 설정
st.set_page_config(page_title='기업 재무 분석 대시보드', layout='wide')
st.title('📊 기업 재무 분석 대시보드')
st.markdown('사업자번호 또는 기업명을 입력하여 기업을 검색하고 성장성 및 수익성 시계열을 확인하세요.')

# 설치된 Nanum 폰트 경로 확인
font_dirs = ['/usr/share/fonts/truetype/nanum/']
font_files = fm.findSystemFonts(fontpaths=font_dirs)

# 폰트 매니저에 폰트 추가
for font_file in font_files:
    fm.fontManager.addfont(font_file)

# 폰트 이름 확인
nanum_font = fm.FontProperties(fname=font_files[0]).get_name()

# 폰트 설정
plt.rcParams['font.family'] = nanum_font
plt.rcParams['axes.unicode_minus'] = False

# ✅ DuckDB 연결 및 데이터 로딩
@st.cache_data
def load_data():
    con = duckdb.connect(database='company_data.duckdb', read_only=False)
    df = con.execute("SELECT * FROM company_data").df()
    con.close()
    return df

company_data = load_data()

# ✅ 사용자 입력: 사업자번호 또는 기업명 검색
search_input = st.text_input('🔍 사업자번호 또는 기업명을 입력하세요')

if search_input:
    # ✅ 필터링된 데이터
    filtered_data = company_data[
        company_data['사업자번호'].astype(str).str.contains(search_input) | 
        company_data['기업명'].str.contains(search_input, na=False)
    ].drop_duplicates(subset=['사업자번호'])
    
    if not filtered_data.empty:
        st.write(f"✅ {len(filtered_data)}개의 기업이 검색되었습니다.")
        # ✅ 기업명 리스트로 selectbox 생성
        selected_company_name = st.selectbox('✅ 분석할 기업명을 선택하세요:', filtered_data['기업명'].unique())
        
        # ✅ 선택한 기업명에 해당하는 사업자번호 찾기
        selected_biz_no = filtered_data[filtered_data['기업명'] == selected_company_name]['사업자번호'].values[0]
        
        # ✅ 선택한 기업 데이터 필터링
        company_df = company_data[company_data['사업자번호'] == selected_biz_no].sort_values('기준연도')
        
        # ✅ 선택한 기업명 및 사업자번호 표시
        selected_company_name = company_df['기업명'].iloc[0] if not company_df.empty else '정보 없음'
        selected_industry = company_df['업종'].iloc[0] if not company_df.empty else '정보 없음'
        st.markdown(f"### 📌 선택된 기업: {selected_company_name} (사업자번호: {selected_biz_no})")
        st.markdown(f"### 🔹 소속 업종: {selected_industry}")
        
        # ✅ 업종 평균 계산용 데이터 필터링
        industry_df = company_data[company_data['업종'] == selected_industry].groupby('기준연도').mean(numeric_only=True).reset_index()
        
        # ✅ 차트 그로로 배치
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('📈 매출 증가율 추이')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(company_df['기준연도'], company_df['매출증가율'], marker='o', linestyle='-', color='skyblue', label='선택 기업')
            ax.plot(industry_df['기준연도'], industry_df['매출증가율'], marker='x', linestyle='--', color='gray', label='업종 평균')
            ax.set_title("매출 증가율 (%)")
            ax.set_xlabel('기준연도')
            ax.set_ylabel('매출 증가율 (%)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            st.subheader('📈 영업이익률 추이')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(company_df['기준연도'], company_df['영업이익률'], marker='o', linestyle='-', color='salmon', label='선택 기업')
            ax.plot(industry_df['기준연도'], industry_df['영업이익률'], marker='x', linestyle='--', color='gray', label='업종 평균')
            ax.set_title("영업이익률 (%)")
            ax.set_xlabel('기준연도')
            ax.set_ylabel('영업이익률 (%)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # ✅ 추가 지표: ROE 및 부채비율 시계열
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader('📈 ROE (자기자본이익률) 추이')
            fig, ax = plt.subplots(figsize=(6, 4))
            company_df['ROE'] = (company_df['손익계산서_당기순이익'] / company_df['재무상태표_자본총계']) * 100
            industry_df['ROE'] = (industry_df['손익계산서_당기순이익'] / industry_df['재무상태표_자본총계']) * 100
            ax.plot(company_df['기준연도'], company_df['ROE'], marker='o', linestyle='-', color='green', label='선택 기업')
            ax.plot(industry_df['기준연도'], industry_df['ROE'], marker='x', linestyle='--', color='gray', label='업종 평균')
            ax.set_title("ROE (%)")
            ax.set_xlabel('기준연도')
            ax.set_ylabel('ROE (%)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        with col4:
            st.subheader('📈 부채비율 추이')
            fig, ax = plt.subplots(figsize=(6, 4))
            company_df['부채비율'] = (company_df['재무상태표_부채총계'] / company_df['재무상태표_자본총계']) * 100
            industry_df['부채비율'] = (industry_df['재무상태표_부채총계'] / industry_df['재무상태표_자본총계']) * 100
            ax.plot(company_df['기준연도'], company_df['부채비율'], marker='o', linestyle='-', color='purple', label='선택 기업')
            ax.plot(industry_df['기준연도'], industry_df['부채비율'], marker='x', linestyle='--', color='gray', label='업종 평균')
            ax.set_title("부채비율 (%)")
            ax.set_xlabel('기준연도')
            ax.set_ylabel('부채비율 (%)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # ✅ 데이터 테이블 표시
        st.subheader('📋 기업 데이터 테이블')
        st.dataframe(company_df)
        
        st.success('✅ 대시보드가 성공적으로 로드되었습니다!')
    else:
        st.warning("검색 결과가 없습니다. 다시 입력해주세요.")
else:
    st.info("사업자번호 또는 기업명을 입력해주세요.")