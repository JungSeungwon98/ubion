import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D,
    BatchNormalization, Flatten, concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging
import os
from datetime import datetime

class NeuralModel:
    def __init__(self, n_splits: int = 5, sequence_length: int = 20):
        """
        주식 가격 예측을 위한 신경망 모델 클래스
        
        Parameters:
        -----------
        n_splits : int
            시계열 교차 검증을 위한 분할 수
        sequence_length : int
            시계열 시퀀스 길이
        """
        self.models = {}
        self.scalers = {}
        self.n_splits = n_splits
        self.sequence_length = sequence_length
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # GPU 메모리 설정
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                self.logger.error(f"GPU memory configuration error: {str(e)}")
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 준비"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 전처리"""
        # 특성 스케일링
        feature_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X)
        self.scalers['feature_scaler'] = feature_scaler
        
        # 타겟 스케일링
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
        self.scalers['target_scaler'] = target_scaler
        
        # 시계열 시퀀스 생성
        X_seq, y_seq = self.prepare_sequences(X_scaled, y_scaled)
        
        return X_seq, y_seq
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """CNN-LSTM 하이브리드 모델 구축"""
        # CNN 브랜치
        cnn_input = Input(shape=input_shape)
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_input)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(filters=128, kernel_size=3, activation='relu')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        
        # LSTM 브랜치
        lstm = LSTM(128, return_sequences=True)(cnn)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(0.2)(lstm)
        lstm = LSTM(64)(lstm)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(0.2)(lstm)
        
        # 완전연결층
        dense = Dense(32, activation='relu')(lstm)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.1)(dense)
        output = Dense(1)(dense)
        
        model = Model(inputs=cnn_input, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_models(self, X: pd.DataFrame, y: pd.Series,
                    models_config: Dict[str, dict] = None) -> Dict[str, dict]:
        """여러 신경망 모델 학습"""
        X_seq, y_seq = self.prepare_data(X, y)
        input_shape = (self.sequence_length, X.shape[1])
        
        if models_config is None:
            models_config = {
                'lstm': {
                    'model': self.build_lstm_model(input_shape)
                },
                'cnn_lstm': {
                    'model': self.build_cnn_lstm_model(input_shape)
                }
            }
        
        results = {}
        
        for name, config in models_config.items():
            try:
                self.logger.info(f"Training {name} model...")
                model = config['model']
                
                # 시계열 교차 검증
                cv_scores = {
                    'mse': [], 'rmse': [], 'mae': [], 'r2': []
                }
                
                for train_idx, val_idx in self.cv.split(X_seq):
                    X_train, X_val = X_seq[train_idx], X_seq[val_idx]
                    y_train, y_val = y_seq[train_idx], y_seq[val_idx]
                    
                    # 조기 종료 콜백
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                    
                    # 모델 학습
                    model.fit(
                        X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # 예측
                    y_pred = model.predict(X_val, verbose=0)
                    
                    # 스케일 역변환
                    y_val_orig = self.scalers['target_scaler'].inverse_transform(y_val.reshape(-1, 1)).ravel()
                    y_pred_orig = self.scalers['target_scaler'].inverse_transform(y_pred).ravel()
                    
                    # 성능 평가
                    cv_scores['mse'].append(mean_squared_error(y_val_orig, y_pred_orig))
                    cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)))
                    cv_scores['mae'].append(mean_absolute_error(y_val_orig, y_pred_orig))
                    cv_scores['r2'].append(r2_score(y_val_orig, y_pred_orig))
                
                # 전체 데이터로 최종 모델 학습
                early_stopping = EarlyStopping(
                    monitor='loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                model.fit(
                    X_seq, y_seq,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # 결과 저장
                results[name] = {
                    'model': model,
                    'cv_scores': {
                        metric: {
                            'mean': np.mean(scores),
                            'std': np.std(scores)
                        }
                        for metric, scores in cv_scores.items()
                    }
                }
                
                self.models[name] = results[name]
                
                self.logger.info(f"{name} model training completed.")
                self.logger.info(f"Average CV scores:")
                for metric, scores in results[name]['cv_scores'].items():
                    self.logger.info(f"{metric}: {scores['mean']:.4f} (+/- {scores['std']:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error training {name} model: {str(e)}")
        
        return results
    
    def predict(self, X: pd.DataFrame, model_name: str = 'lstm') -> np.ndarray:
        """예측 수행"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Please train the model first.")
        
        try:
            # 특성 스케일링
            X_scaled = self.scalers['feature_scaler'].transform(X)
            
            # 시퀀스 생성
            X_seq = []
            for i in range(len(X_scaled) - self.sequence_length):
                X_seq.append(X_scaled[i:(i + self.sequence_length)])
            X_seq = np.array(X_seq)
            
            # 예측
            y_pred_scaled = self.models[model_name]['model'].predict(X_seq, verbose=0)
            
            # 스케일 역변환
            y_pred = self.scalers['target_scaler'].inverse_transform(y_pred_scaled)
            
            return y_pred.ravel()
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def save_models(self, directory: str = 'models/saved'):
        """모델 저장"""
        try:
            os.makedirs(directory, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for name, model_info in self.models.items():
                model_path = os.path.join(directory, f"{name}_model_{timestamp}")
                scaler_path = os.path.join(directory, f"{name}_scalers_{timestamp}.npz")
                
                # 모델 저장
                model_info['model'].save(model_path)
                
                # 스케일러 저장
                np.savez(
                    scaler_path,
                    feature_scaler=self.scalers['feature_scaler'],
                    target_scaler=self.scalers['target_scaler']
                )
                
                self.logger.info(f"Saved {name} model to {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, model_path: str, scaler_path: str):
        """모델 로드"""
        try:
            model = tf.keras.models.load_model(model_path)
            scalers = np.load(scaler_path)
            
            model_name = os.path.basename(model_path).split('_')[0]
            self.models[model_name] = {'model': model}
            self.scalers = {
                'feature_scaler': scalers['feature_scaler'].item(),
                'target_scaler': scalers['target_scaler'].item()
            }
            
            self.logger.info(f"Loaded model from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, dict]:
        """모델 학습 - train_models 메서드의 래퍼"""
        self.logger.info("Training neural models...")
        return self.train_models(X, y)

def main():
    # 예제 사용법
    import yfinance as yf
    from ..data.feature_engineering import FeatureEngineer
    
    # 데이터 가져오기
    symbol = 'AAPL'
    stock = yf.Ticker(symbol)
    data = stock.history(start='2020-01-01')
    
    # 특성 엔지니어링
    engineer = FeatureEngineer(data)
    data = engineer.add_technical_indicators()
    data = engineer.add_time_features()
    data = engineer.add_lagged_features()
    
    # 특성과 타겟 준비
    feature_columns = [col for col in data.columns if col not in ['Close']]
    X = data[feature_columns].dropna()
    y = data['Close'].loc[X.index]
    
    # 모델 학습
    models = StockNeuralModels(n_splits=5, sequence_length=20)
    results = models.train_models(X, y)
    
    # 모델 저장
    models.save_models()

if __name__ == "__main__":
    main() 