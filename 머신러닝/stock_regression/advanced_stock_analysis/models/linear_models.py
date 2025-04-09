import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import joblib
from datetime import datetime
import os
from sklearn.impute import SimpleImputer

class LinearModel:
    def __init__(self, model_type: str = 'linear', n_splits: int = 5):
        """
        주식 가격 예측을 위한 선형 모델 클래스
        
        Parameters:
        -----------
        model_type : str
            사용할 모델 유형 ('linear', 'ridge', 'lasso', 'elastic_net')
        n_splits : int
            시계열 교차 검증을 위한 분할 수
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.n_splits = n_splits
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _get_model(self):
        """모델 유형에 따라 적절한 모델 반환"""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'lasso':
            return Lasso(alpha=0.1, random_state=42)
        elif self.model_type == 'elastic_net':
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, X: pd.DataFrame) -> np.ndarray:
        """데이터 전처리: NaN 값 처리 후 스케일링"""
        # NaN 값 확인
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values in X, applying imputation")
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = X.values
        
        # 스케일링
        X_scaled = self.scaler.fit_transform(X_imputed)
        return X_scaled
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """모델 학습"""
        try:
            self.logger.info(f"Training {self.model_type} model...")
            
            # NaN 값 확인 및 처리
            nan_count_X = X.isna().sum().sum()
            nan_count_y = y.isna().sum()
            
            if nan_count_X > 0 or nan_count_y > 0:
                self.logger.warning(f"Found NaN values in input data: X={nan_count_X}, y={nan_count_y}")
                
                # X 데이터의 NaN 처리
                if nan_count_X > 0:
                    self.logger.info("Imputing NaN values in X")
                    X_imputed = pd.DataFrame(
                        self.imputer.fit_transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                else:
                    X_imputed = X
                
                # y 데이터의 NaN 처리
                if nan_count_y > 0:
                    self.logger.info("Imputing NaN values in y")
                    y_imputer = SimpleImputer(strategy='mean')
                    y_imputed = pd.Series(
                        y_imputer.fit_transform(y.values.reshape(-1, 1)).flatten(),
                        index=y.index,
                        name=y.name
                    )
                else:
                    y_imputed = y
                
                X = X_imputed
                y = y_imputed
            
            # 데이터 전처리
            X_scaled = self.scaler.fit_transform(X)
            
            # 모델 초기화
            self.model = self._get_model()
            
            # 시계열 교차 검증
            cv_scores = {
                'mse': [], 'rmse': [], 'mae': [], 'r2': []
            }
            
            for train_idx, val_idx in self.cv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 모델 학습
                self.model.fit(X_train, y_train)
                
                # 예측
                y_pred = self.model.predict(X_val)
                
                # 성능 평가
                cv_scores['mse'].append(mean_squared_error(y_val, y_pred))
                cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
                cv_scores['mae'].append(mean_absolute_error(y_val, y_pred))
                cv_scores['r2'].append(r2_score(y_val, y_pred))
            
            # 전체 데이터로 최종 모델 학습
            self.model.fit(X_scaled, y)
            
            # 결과 반환
            return {
                'cv_scores': {
                    metric: {
                        'mean': np.mean(scores),
                        'std': np.std(scores)
                    }
                    for metric, scores in cv_scores.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측 수행: NaN 값 처리 후 예측"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        try:
            # NaN 값 확인 및 처리
            nan_count = X.isna().sum().sum()
            if nan_count > 0:
                self.logger.warning(f"Found {nan_count} NaN values in prediction data, applying imputation")
                # 열별로 처리
                X_clean = X.copy()
                for col in X_clean.columns:
                    if X_clean[col].isna().any():
                        X_clean[col] = X_clean[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            else:
                X_clean = X
            
            # 스케일링 및 예측
            X_scaled = self.scaler.transform(X_clean)
            return self.model.predict(X_scaled)
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            # 기본 특성 예측 재시도
            try:
                self.logger.info("Retrying prediction with minimal features")
                # 모델이 훈련된 특성 확인
                if hasattr(self.model, 'feature_names_in_'):
                    model_features = self.model.feature_names_in_
                    self.logger.info(f"Model was trained on features: {model_features}")
                    
                    # 공통 특성만 사용
                    common_features = [col for col in X.columns if col in model_features]
                    if common_features:
                        self.logger.info(f"Using common features: {common_features}")
                        X_subset = X[common_features].fillna(0)
                        X_scaled = self.scaler.transform(X_subset)
                        return self.model.predict(X_scaled)
                
                # 간단하게 재시도
                self.logger.warning("Fallback to fixed imputation")
                X_simple = X.fillna(0).values
                X_scaled = self.scaler.transform(X_simple)
                return self.model.predict(X_scaled)
                
            except Exception as e2:
                self.logger.error(f"Prediction retry failed: {str(e2)}")
                # 원본 오류 전달
                raise e
    
    def get_feature_names(self) -> List[str]:
        """모델 훈련에 사용된 특성 이름 반환"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        else:
            self.logger.warning("Model does not have feature_names_in_ attribute")
            return []
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """특성 중요도 계산 (가능한 경우)"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # 모델 유형에 따라 계수 추출 (조정 필요)
        if hasattr(self.model, 'coef_'):
            # 절대값으로 중요도 계산
            importance = np.abs(self.model.coef_)
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            self.logger.warning("This model type does not support feature importance")
            return pd.DataFrame()
    
    def save_model(self, directory: str = 'models/saved'):
        """모델 저장"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 모델 저장
        model_path = os.path.join(directory, f"{self.model_type}_{timestamp}.joblib")
        joblib.dump(self.model, model_path)
        
        # 스케일러 및 imputer 저장
        scaler_path = os.path.join(directory, f"{self.model_type}_scaler_{timestamp}.joblib")
        imputer_path = os.path.join(directory, f"{self.model_type}_imputer_{timestamp}.joblib")
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.imputer, imputer_path)
        
        self.logger.info(f"Saved model to {model_path}")
        self.logger.info(f"Saved scaler to {scaler_path}")
        self.logger.info(f"Saved imputer to {imputer_path}")
        
        return model_path, scaler_path, imputer_path
    
    def load_model(self, model_path: str, scaler_path: str, imputer_path: Optional[str] = None):
        """모델 로드"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            if imputer_path and os.path.exists(imputer_path):
                self.imputer = joblib.load(imputer_path)
            
            self.logger.info(f"Loaded model from {model_path}")
            self.logger.info(f"Loaded scaler from {scaler_path}")
            if imputer_path:
                self.logger.info(f"Loaded imputer from {imputer_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    # 예제 사용법
    from data.data_loader import DataLoader
    
    # 데이터 로드
    loader = DataLoader()
    data = loader.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # 특성 및 타겟 설정
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_col = 'Returns'
    
    # 데이터 분할
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # 모델 학습
    model = LinearModel(model_type='ridge')
    results = model.train(train_data[feature_cols], train_data[target_col])
    
    # 예측
    predictions = model.predict(test_data[feature_cols])
    
    # 특성 중요도
    importance = model.get_feature_importance(feature_cols)
    print(importance)
    
    # 모델 저장
    model.save_model()

if __name__ == "__main__":
    main() 