import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import glob
from data_loader import DataLoader
from stock_regression import StockRegressor

# 페이지 설정
st.set_page_config(
    page_title="주식 가격 예측 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("<h1 class='main-header'>주식 가격 예측 대시보드</h1>", unsafe_allow_html=True)

# 사이드바 설정
st.sidebar.header("설정")

# 주식 심볼 선택
symbol = st.sidebar.text_input("주식 심볼", value="AAPL")
st.sidebar.markdown("예시: AAPL, MSFT, GOOGL, TSLA, AMZN")

# 날짜 범위 선택
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "시작일",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "종료일",
        value=datetime.now(),
        max_value=datetime.now()
    )

# 모델 선택
model_type = st.sidebar.selectbox(
    "회귀 모델",
    options=["linear", "ridge", "lasso", "elasticnet"],
    index=0
)

# 모델 파라미터
if model_type in ["ridge", "lasso", "elasticnet"]:
    alpha = st.sidebar.slider("알파 (정규화 강도)", 0.01, 10.0, 1.0, 0.01)
else:
    alpha = 1.0

if model_type == "elasticnet":
    l1_ratio = st.sidebar.slider("L1 비율", 0.0, 1.0, 0.5, 0.01)
else:
    l1_ratio = 0.5

# 테스트 세트 크기
test_size = st.sidebar.slider("테스트 세트 크기", 0.1, 0.5, 0.2, 0.05)

# 분석 실행 버튼
run_analysis = st.sidebar.button("분석 실행")

# 메인 대시보드
if run_analysis:
    # 진행 상태 표시
    with st.spinner("데이터를 가져오고 모델을 학습하는 중..."):
        # 데이터 로더 초기화
        loader = DataLoader()
        
        # 데이터 가져오기
        data = loader.fetch_stock_data(
            symbol, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        if data is None or data.empty:
            st.error(f"{symbol}에 대한 데이터를 가져오는데 실패했습니다. 심볼이나 날짜 범위를 확인해주세요.")
        else:
            # 회귀 모델 초기화
            regressor = StockRegressor()
            
            # 데이터 준비
            X_train, X_test, y_train, y_test = regressor.prepare_data(
                data, test_size=test_size
            )
            
            if X_train is None:
                st.error("데이터 준비에 실패했습니다. 입력 매개변수를 확인해주세요.")
            else:
                # 모델 파라미터 설정
                model_params = {}
                if model_type in ['ridge', 'lasso', 'elasticnet']:
                    model_params['alpha'] = alpha
                if model_type == 'elasticnet':
                    model_params['l1_ratio'] = l1_ratio
                
                # 모델 학습
                regressor.train_model(X_train, y_train, model_type=model_type, **model_params)
                
                # 모델 평가
                y_pred, results = regressor.evaluate_model(X_test, y_test)
                
                if y_pred is None:
                    st.error("모델 평가에 실패했습니다. 입력 매개변수를 확인해주세요.")
                else:
                    # 결과 표시
                    st.markdown(f"<h2 class='sub-header'>{symbol} 분석 결과</h2>", unsafe_allow_html=True)
                    
                    # 메트릭 카드
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-value'>{results['RMSE']:.2f}</div>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-label'>평균 제곱근 오차 (RMSE)</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-value'>{results['R2']:.4f}</div>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-label'>결정계수 (R²)</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-value'>{results['MAPE']:.2f}%</div>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-label'>평균 절대 백분율 오차 (MAPE)</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # 예측 결과 시각화
                    st.markdown("<h3 class='sub-header'>예측 결과</h3>", unsafe_allow_html=True)
                    
                    # Plotly를 사용한 인터랙티브 차트
                    fig = go.Figure()
                    
                    # 실제 가격
                    fig.add_trace(go.Scatter(
                        x=y_test.index,
                        y=y_test.values,
                        mode='lines',
                        name='실제 가격',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # 예측 가격
                    fig.add_trace(go.Scatter(
                        x=y_test.index,
                        y=y_pred,
                        mode='lines',
                        name='예측 가격',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} 주식 가격 예측 ({regressor.model_name} 모델)",
                        xaxis_title="날짜",
                        yaxis_title="가격",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 잔차 분석
                    st.markdown("<h3 class='sub-header'>잔차 분석</h3>", unsafe_allow_html=True)
                    
                    # 잔차 계산
                    residuals = y_test - y_pred
                    
                    # 잔차 플롯
                    fig_res = go.Figure()
                    
                    fig_res.add_trace(go.Scatter(
                        x=y_pred,
                        y=residuals,
                        mode='markers',
                        name='잔차',
                        marker=dict(
                            color='rgba(0, 0, 255, 0.5)',
                            size=8
                        )
                    ))
                    
                    # 0선 추가
                    fig_res.add_trace(go.Scatter(
                        x=[min(y_pred), max(y_pred)],
                        y=[0, 0],
                        mode='lines',
                        name='기준선',
                        line=dict(color='red', width=1, dash='dash')
                    ))
                    
                    fig_res.update_layout(
                        title=f"{regressor.model_name} 모델의 잔차 분석",
                        xaxis_title="예측값",
                        yaxis_title="잔차",
                        height=400
                    )
                    
                    st.plotly_chart(fig_res, use_container_width=True)
                    
                    # 특성 중요도
                    st.markdown("<h3 class='sub-header'>특성 중요도</h3>", unsafe_allow_html=True)
                    
                    if hasattr(regressor.model, 'coef_'):
                        # 특성 중요도 계산
                        importance = np.abs(regressor.model.coef_)
                        
                        # 특성 중요도 데이터프레임
                        feature_importance = pd.DataFrame({
                            '특성': regressor.feature_columns,
                            '중요도': importance
                        })
                        feature_importance = feature_importance.sort_values('중요도', ascending=True)
                        
                        # 특성 중요도 플롯
                        fig_imp = go.Figure(go.Bar(
                            x=feature_importance['중요도'],
                            y=feature_importance['특성'],
                            orientation='h',
                            marker_color='rgba(30, 136, 229, 0.7)'
                        ))
                        
                        fig_imp.update_layout(
                            title=f"{regressor.model_name} 모델의 특성 중요도",
                            xaxis_title="계수 절대값",
                            yaxis_title="특성",
                            height=500
                        )
                        
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.warning("이 모델 유형에서는 특성 중요도를 확인할 수 없습니다.")
                    
                    # 모델 평가 지표 테이블
                    st.markdown("<h3 class='sub-header'>모델 평가 지표</h3>", unsafe_allow_html=True)
                    
                    metrics_df = pd.DataFrame({
                        '지표': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²', '설명된 분산'],
                        '값': [
                            f"{results['MSE']:.2f}",
                            f"{results['RMSE']:.2f}",
                            f"{results['MAE']:.2f}",
                            f"{results['MAPE']:.2f}%",
                            f"{results['R2']:.4f}",
                            f"{results['Explained Variance']:.4f}"
                        ]
                    })
                    
                    st.table(metrics_df)
                    
                    # 원본 데이터 표시
                    st.markdown("<h3 class='sub-header'>역사적 데이터</h3>", unsafe_allow_html=True)
                    
                    st.dataframe(data.head(10))
                    
                    # 다운로드 버튼
                    csv = data.to_csv(index=True)
                    st.download_button(
                        label="CSV 파일로 다운로드",
                        data=csv,
                        file_name=f"{symbol}_data.csv",
                        mime="text/csv"
                    )

# 사이드바 하단에 정보 추가
st.sidebar.markdown("---")
st.sidebar.markdown("### 대시보드 소개")
st.sidebar.markdown("""
이 대시보드에서 할 수 있는 것:
- 주식 가격 데이터 분석
- 회귀 모델 학습
- 예측 결과 시각화
- 모델 성능 평가
""")

# 푸터
st.markdown("---")
st.markdown("주식 가격 예측 대시보드 | Streamlit으로 제작") 