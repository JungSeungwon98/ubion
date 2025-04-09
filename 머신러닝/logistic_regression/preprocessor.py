import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import urllib.parse
import traceback
import streamlit as st

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """데이터를 전처리하고 학습/테스트 세트로 분할합니다."""
        try:
            # 'target' 컬럼이 있는지 확인
            if 'target' in features.columns:
                # 특성과 타겟 분리
                X = features.drop('target', axis=1)
                y = features['target']
            else:
                # 'target' 컬럼이 없는 경우, 모든 컬럼을 특성으로 사용
                X = features.copy()
                # 임시 타겟 생성 (실제로는 사용되지 않음)
                y = pd.Series(np.zeros(len(features)))
                st.warning("'target' 컬럼이 없습니다. 모든 컬럼을 특성으로 사용합니다.")
            
            # 데이터 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # 학습/테스트 세트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            error_msg = str(e)
            st.error(f"데이터 전처리 중 오류가 발생했습니다: {error_msg}")
            
            # ChatGPT 버튼 생성
            error_text = f"Streamlit 앱에서 다음 오류가 발생했습니다: {error_msg}" + "\n\n스택 트레이스:\n" + traceback.format_exc()
            chatgpt_url = f"https://chat.openai.com/chat?message={urllib.parse.quote(error_text)}"
            st.markdown(f'<a href="{chatgpt_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;">ChatGPT에게 이 오류에 대해 물어보기</button></a>', unsafe_allow_html=True)
            
            # 오류 발생 시 빈 배열 반환
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def transform_new_data(self, new_data: pd.DataFrame) -> np.ndarray:
        """새로운 데이터를 변환합니다."""
        return self.scaler.transform(new_data)
    
    def get_feature_names(self) -> list:
        """특성 이름 목록을 반환합니다."""
        feature_names = [
            'Returns',
            'Volatility',
            'Volume_Change',
            'Volume_MA',
            'VIX_Change',
            'VIX_MA'
        ]
        return feature_names

if __name__ == "__main__":
    # 테스트 코드
    from data_loader import MarketDataLoader
    from datetime import datetime, timedelta
    
    loader = MarketDataLoader()
    data = loader.load_market_data("AAPL")
    features = loader.prepare_features(
        data
    )
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(features)
    
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print(X_train)

    try:
        # 분석 시작 버튼을 클릭했을 때의 로직
        pass
    except Exception as e:
        error_msg = str(e)
        st.error(f"오류가 발생했습니다: {error_msg}")
        
        # ChatGPT 버튼 생성
        error_text = f"Streamlit 앱에서 다음 오류가 발생했습니다: {error_msg}" + "\n\n스택 트레이스:\n" + traceback.format_exc()
        chatgpt_url = f"https://chat.openai.com/chat?message={urllib.parse.quote(error_text)}"
        st.markdown(f'<a href="{chatgpt_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;">ChatGPT에게 이 오류에 대해 물어보기</button></a>', unsafe_allow_html=True) 