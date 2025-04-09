import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
import time
import urllib.parse
import traceback


from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor
from model import VolatilityPredictor
from report_generator import ReportGenerator

# 페이지 설정
st.set_page_config(
    page_title="주식 시장 변동성 예측 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #757575;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<h1 class="main-header">주식 시장 변동성 예측 대시보드</h1>', unsafe_allow_html=True)

# 사이드바 설정
st.sidebar.title("설정")
symbol = st.sidebar.text_input("주식 심볼", value="AAPL")
days = st.sidebar.slider("분석 기간 (일)", min_value=30, max_value=365, value=180)

# 기간 설정
today = datetime.now()
default_start = today - timedelta(days=365)
start_date = st.sidebar.date_input("시작일", value=default_start)
end_date = st.sidebar.date_input("종료일", value=today)

# 분석 버튼
analyze_button = st.sidebar.button("분석 실행", key="analyze_button")

# 사이드바에 정보 추가
st.sidebar.markdown("---")
st.sidebar.markdown("### 정보")
st.sidebar.info(
    "이 대시보드는 로지스틱 회귀를 사용하여 주식 시장의 변동성 급증을 예측합니다. "
    "주가 데이터, VIX 지수, 거래량 등의 정보를 활용하여 다음 날의 변동성이 현재보다 "
    "20% 이상 높을지 예측합니다."
)

# 메인 콘텐츠
if analyze_button:
    with st.spinner("데이터를 로드하고 분석 중입니다..."):
        # 진행 상태 표시
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        try:
            # 데이터 로드
            loader = MarketDataLoader()
            data = loader.load_market_data(symbol, days)
            
            try:
                features = loader.prepare_features(data)
            except Exception as e:
                error_msg = str(e)
                st.error(f"특성 준비 중 오류가 발생했습니다: {error_msg}")
                
                # ChatGPT 버튼 생성
                error_text = f"Streamlit 앱에서 다음 오류가 발생했습니다: {error_msg}" + "\n\n스택 트레이스:\n" + traceback.format_exc()
                chatgpt_url = f"https://chat.openai.com/chat?message={urllib.parse.quote(error_text)}"
                st.markdown(f'<a href="{chatgpt_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;">ChatGPT에게 이 오류에 대해 물어보기</button></a>', unsafe_allow_html=True)
                
                st.stop()
            
            # 데이터 전처리
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.prepare_data(features)
            
            # 데이터가 비어있는지 확인
            if len(X_train) == 0 or len(X_test) == 0:
                st.error("데이터 전처리 중 오류가 발생했습니다. 다른 주식 심볼이나 기간을 선택해보세요.")
                st.stop()
            
            # 모델 학습
            predictor = VolatilityPredictor()
            predictor.train(X_train, y_train)
            
            # 모델 평가
            eval_results = predictor.evaluate(X_test, y_test)
            
            # 예측 수행
            predictions = predictor.predict(X_test)
            probabilities = predictor.predict_proba(X_test)[:, 1]
            
            # 테스트 데이터의 인덱스 가져오기
            test_indices = features.index[-len(X_test):]
            
            # 결과 표시
            st.markdown('<h2 class="sub-header">분석 결과</h2>', unsafe_allow_html=True)
            
            # 주요 지표 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("정확도", f"{float(eval_results['accuracy']):.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("데이터 기간", f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("데이터 포인트", f"{len(features)}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 차트 표시
            st.markdown('<h2 class="sub-header">변동성 분석</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3>주가 변동성 추이</h3>', unsafe_allow_html=True)
                fig_volatility = go.Figure()
                fig_volatility.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Volatility'],
                    name='변동성',
                    line=dict(color='blue')
                ))
                fig_volatility.update_layout(
                    title='주가 변동성 추이',
                    xaxis_title='날짜',
                    yaxis_title='변동성',
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_volatility, use_container_width=True)
                
                # 주가 데이터 표시
                st.markdown('<h3>주가 데이터</h3>', unsafe_allow_html=True)
                st.dataframe(data.head(10), use_container_width=True)
            
            with col2:
                st.markdown('<h3>변동성 급증 예측 확률</h3>', unsafe_allow_html=True)
                fig_prediction = go.Figure()
                fig_prediction.add_trace(go.Scatter(
                    x=test_indices,
                    y=probabilities,
                    name='변동성 급증 확률',
                    line=dict(color='red')
                ))
                fig_prediction.update_layout(
                    title='변동성 급증 예측 확률',
                    xaxis_title='날짜',
                    yaxis_title='확률',
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_prediction, use_container_width=True)
                
                # 특징 데이터 표시
                st.markdown('<h3>특징 데이터</h3>', unsafe_allow_html=True)
                st.dataframe(features.head(10), use_container_width=True)
            
            # 특성 중요도 및 모델 성능
            st.markdown('<h2 class="sub-header">모델 분석</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3>특성 중요도</h3>', unsafe_allow_html=True)
                importance = predictor.get_feature_importance(preprocessor.get_feature_names())
                
                if not importance.empty:
                    fig_importance = px.bar(
                        importance,
                        x='feature',
                        y='importance',
                        title='특성 중요도',
                        color='importance',
                        color_continuous_scale='Viridis'
                    )
                    fig_importance.update_layout(
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.warning("특성 중요도를 계산할 수 없습니다. 데이터를 확인해주세요.")
            
            with col2:
                st.markdown('<h3>모델 성능 지표</h3>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("**분류 보고서:**")
                st.text(eval_results['classification_report'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 데이터 다운로드
            st.markdown('<h2 class="sub-header">데이터 다운로드</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 특징 데이터 다운로드
                features_csv = features.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="특징 데이터 다운로드 (CSV)",
                    data=features_csv,
                    file_name=f"{symbol}_features.csv",
                    mime="text/csv"
                )
            
            with col2:
                # 예측 결과 다운로드
                results_df = pd.DataFrame({
                    'Date': test_indices,
                    'Volatility': data.loc[test_indices, 'Volatility'],
                    'Prediction': predictions,
                    'Probability': probabilities
                })
                results_csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="예측 결과 다운로드 (CSV)",
                    data=results_csv,
                    file_name=f"{symbol}_predictions.csv",
                    mime="text/csv"
                )
            
            # 모델 저장 정보
            model_path = f'model_{symbol}.joblib'
            predictor.save_model(model_path)
            st.sidebar.success(f"모델이 '{model_path}'로 저장되었습니다.")
            
            # PDF 보고서 생성 버튼
            if st.button("PDF 보고서 생성"):
                try:
                    # reports 디렉토리 생성
                    os.makedirs('reports', exist_ok=True)
                    
                    # 보고서 파일명 생성
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    report_path = f'reports/{symbol}_report_{timestamp}.pdf'
                    
                    # 보고서 생성
                    generator = ReportGenerator(
                        symbol,
                        data,
                        features,
                        predictions,
                        probabilities,
                        importance,
                        eval_results
                    )
                    
                    report_path = generator.generate_report(report_path)
                    
                    # 다운로드 버튼 생성
                    with open(report_path, 'rb') as f:
                        st.download_button(
                            label="PDF 보고서 다운로드",
                            data=f,
                            file_name=os.path.basename(report_path),
                            mime='application/pdf'
                        )
                    
                    st.success("PDF 보고서가 생성되었습니다!")
                    
                except Exception as e:
                    st.error(f"PDF 보고서 생성 중 오류가 발생했습니다: {str(e)}")
                    error_message = str(e)
                    error_traceback = st.exception(e)
                    
                    # ChatGPT에 질문하기 버튼
                    encoded_error = error_message.replace(" ", "+")
                    chatgpt_url = f"https://chat.openai.com/chat?message={encoded_error}"
                    st.markdown(f"[ChatGPT에 이 오류에 대해 물어보기]({chatgpt_url})")
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"오류가 발생했습니다: {error_msg}")
            
            # ChatGPT 버튼 생성
            error_text = f"Streamlit 앱에서 다음 오류가 발생했습니다: {error_msg}" + "\n\n스택 트레이스:\n" + traceback.format_exc()
            chatgpt_url = f"https://chat.openai.com/chat?message={urllib.parse.quote(error_text)}"
            st.markdown(f'<a href="{chatgpt_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;">ChatGPT에게 이 오류에 대해 물어보기</button></a>', unsafe_allow_html=True)
            
            st.info("다른 주식 심볼이나 기간을 선택해보세요.")
else:
    # 초기 화면
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ## 환영합니다!
    
    이 대시보드는 주식 시장의 변동성 급증을 예측하는 로지스틱 회귀 모델을 시각화합니다.
    
    ### 사용 방법:
    1. 왼쪽 사이드바에서 주식 심볼과 분석 기간을 설정하세요.
    2. '분석 실행' 버튼을 클릭하여 분석을 시작하세요.
    3. 분석 결과는 자동으로 화면에 표시됩니다.
    
    ### 주요 기능:
    - 주가 변동성 추이 시각화
    - 변동성 급증 예측 확률
    - 특성 중요도 분석
    - 모델 성능 평가
    - 데이터 다운로드
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 예시 차트
    st.markdown('<h2 class="sub-header">예시 차트</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3>주가 변동성 추이 (예시)</h3>', unsafe_allow_html=True)
        # 예시 데이터 생성
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        volatility = np.random.normal(0.02, 0.005, len(dates))
        volatility = volatility + np.sin(np.arange(len(dates)) * 0.1) * 0.01
        
        fig_example = go.Figure()
        fig_example.add_trace(go.Scatter(
            x=dates,
            y=volatility,
            name='변동성',
            line=dict(color='blue')
        ))
        fig_example.update_layout(
            title='주가 변동성 추이 (예시)',
            xaxis_title='날짜',
            yaxis_title='변동성',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_example, use_container_width=True)
    
    with col2:
        st.markdown('<h3>변동성 급증 예측 확률 (예시)</h3>', unsafe_allow_html=True)
        # 예시 데이터 생성
        proba_example = np.random.uniform(0, 1, len(dates))
        proba_example = proba_example + np.sin(np.arange(len(dates)) * 0.05) * 0.3
        proba_example = np.clip(proba_example, 0, 1)
        
        fig_proba_example = go.Figure()
        fig_proba_example.add_trace(go.Scatter(
            x=dates,
            y=proba_example,
            name='변동성 급증 확률',
            line=dict(color='red')
        ))
        fig_proba_example.update_layout(
            title='변동성 급증 예측 확률 (예시)',
            xaxis_title='날짜',
            yaxis_title='확률',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_proba_example, use_container_width=True)

# 푸터
st.markdown('<div class="footer">© 2023 주식 시장 변동성 예측 시스템</div>', unsafe_allow_html=True) 