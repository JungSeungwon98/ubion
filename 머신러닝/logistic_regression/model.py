import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Tuple, Dict

class VolatilityPredictor:
    def __init__(self):
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """모델을 학습시킵니다."""
        self.model.fit(X_train, y_train)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측을 수행합니다."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """확률 예측을 수행합니다."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """모델 성능을 평가합니다."""
        y_pred = self.predict(X_test)
        
        accuracy = float(accuracy_score(y_test, y_pred))
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def save_model(self, filepath: str) -> None:
        """모델을 저장합니다."""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """모델을 로드합니다."""
        self.model = joblib.load(filepath)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """특성 중요도를 반환합니다."""
        try:
            # 특성 이름과 중요도 값의 길이가 같은지 확인
            if len(feature_names) != len(self.model.coef_[0]):
                raise ValueError(f"특성 이름 개수({len(feature_names)})와 특성 중요도 개수({len(self.model.coef_[0])})가 일치하지 않습니다.")
            
            # 특성 중요도 계산
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(self.model.coef_[0])
            })
            return importance.sort_values('importance', ascending=False)
        except Exception as e:
            print(f"특성 중요도 계산 중 오류 발생: {str(e)}")
            # 오류 발생 시 빈 데이터프레임 반환
            return pd.DataFrame(columns=['feature', 'importance'])

if __name__ == "__main__":
    # 테스트 코드
    from data_loader import MarketDataLoader
    from preprocessor import DataPreprocessor
    from datetime import datetime, timedelta
    
    # 데이터 로드
    loader = MarketDataLoader()
    data = loader.load_market_data("AAPL")
    features = loader.prepare_features(
        data
    )
    
    # 데이터 전처리
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(features)
    
    # 모델 학습 및 평가
    predictor = VolatilityPredictor()
    predictor.train(X_train, y_train)
    
    # 평가 결과 출력
    results = predictor.evaluate(X_test, y_test)
    print("Accuracy:", results['accuracy'])
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # 특성 중요도 출력
    importance = predictor.get_feature_importance(preprocessor.get_feature_names())
    print("\nFeature Importance:")
    print(importance) 