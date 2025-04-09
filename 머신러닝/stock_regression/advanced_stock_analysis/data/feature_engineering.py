import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import logging
from datetime import datetime
import pandas_ta as ta  # talib 대신 pandas_ta 사용

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """주식 데이터에 대한 특성 엔지니어링을 수행하는 클래스"""
    
    def __init__(self):
        """특성 엔지니어링 클래스 초기화"""
        self.technical_indicators = []
        logger.info("FeatureEngineer initialized")
    
    def process(self, df):
        """전체 특성 엔지니어링 프로세스 실행"""
        try:
            logger.info("Starting feature engineering process")
            df = self.add_technical_indicators(df)
            df = self.add_time_features(df)
            df = self.add_lagged_features(df)
            df = self.add_rolling_features(df)
            logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error in feature engineering process: {str(e)}")
            raise
    
    def add_technical_indicators(self, df):
        """기술적 지표 추가"""
        try:
            # 데이터프레임 검증
            if df is None:
                logger.error("Input DataFrame is None")
                return pd.DataFrame()
                
            if df.empty:
                logger.error("Input DataFrame is empty")
                return df
                
            # 필요한 컬럼이 있는지 확인
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Available columns: {df.columns.tolist()}")
                return df  # 원본 데이터프레임 반환
            
            logger.info("Adding technical indicators")
            logger.info(f"DataFrame shape before adding indicators: {df.shape}")
            
            # 기본 기술적 지표 (각 단계별로 예외 처리)
            try:
                df['SMA_20'] = ta.sma(df['Close'], length=20)
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['EMA_20'] = ta.ema(df['Close'], length=20)
                logger.info("Added SMA and EMA indicators")
            except Exception as e:
                logger.error(f"Error adding SMA/EMA indicators: {str(e)}")
            
            try:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                logger.info("Added RSI indicator")
            except Exception as e:
                logger.error(f"Error adding RSI indicator: {str(e)}")
            
            try:
                macd = ta.macd(df['Close'])
                if isinstance(macd, pd.DataFrame) and not macd.empty:
                    # MACD 컬럼 이름 확인
                    logger.info(f"MACD columns: {macd.columns.tolist()}")
                    if 'MACD_12_26_9' in macd.columns:
                        df['MACD'] = macd['MACD_12_26_9']
                    elif len(macd.columns) >= 3:
                        df['MACD'] = macd.iloc[:, 0]
                        df['MACD_Signal'] = macd.iloc[:, 1]
                        df['MACD_Hist'] = macd.iloc[:, 2]
                    logger.info("Added MACD indicators")
                else:
                    logger.warning("MACD calculation returned empty result")
            except Exception as e:
                logger.error(f"Error adding MACD indicator: {str(e)}")
            
            try:
                # 볼린저 밴드
                bbands = ta.bbands(df['Close'], length=20, std=2)
                if isinstance(bbands, pd.DataFrame) and not bbands.empty:
                    logger.info(f"Bollinger Bands columns: {bbands.columns.tolist()}")
                    
                    # 볼린저 밴드의 처음 3개 컬럼이 상단, 중간, 하단 밴드
                    if len(bbands.columns) >= 3:
                        df['BB_upper'] = bbands.iloc[:, 0]
                        df['BB_middle'] = bbands.iloc[:, 1]
                        df['BB_lower'] = bbands.iloc[:, 2]
                    else:
                        logger.warning(f"Bollinger Bands returned unexpected format")
                        # 볼린저 밴드 계산 대체
                        df['BB_middle'] = df['SMA_20'] if 'SMA_20' in df.columns else df['Close'].rolling(window=20).mean()
                        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
                        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
                    
                    logger.info("Added Bollinger Bands indicators")
                else:
                    logger.warning("Bollinger Bands calculation returned empty result")
            except Exception as e:
                logger.error(f"Error adding Bollinger Bands indicators: {str(e)}")
            
            try:
                # 거래량 기반 지표
                df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
                
                # On-Balance Volume
                df['OBV'] = ta.obv(df['Close'], df['Volume'])
                logger.info("Added volume-based indicators")
            except Exception as e:
                logger.error(f"Error adding volume-based indicators: {str(e)}")
            
            try:
                # 변동성 지표
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                logger.info("Added ATR indicator")
            except Exception as e:
                logger.error(f"Error adding ATR indicator: {str(e)}")
                
            try:
                # 스토캐스틱 오실레이터
                stoch = ta.stoch(df['High'], df['Low'], df['Close'])
                if isinstance(stoch, pd.DataFrame) and not stoch.empty:
                    logger.info(f"Stochastic columns: {stoch.columns.tolist()}")
                    if len(stoch.columns) >= 2:
                        df['Stoch_K'] = stoch.iloc[:, 0]
                        df['Stoch_D'] = stoch.iloc[:, 1]
                        logger.info("Added Stochastic Oscillator")
                else:
                    logger.warning("Stochastic calculation returned empty result")
            except Exception as e:
                logger.error(f"Error adding Stochastic Oscillator: {str(e)}")
            
            # NaN 값 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Technical indicators added successfully. DataFrame shape after: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            # 오류가 발생해도 원본 데이터프레임 반환
            return df
    
    def add_time_features(self, df):
        """시간 기반 특성 추가"""
        try:
            logger.info("Adding time features")
            
            # 인덱스가 datetime 형식인지 확인
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # 시간 특성 추출
            df['Year'] = df.index.year
            df['Month'] = df.index.month
            df['Day'] = df.index.day
            df['DayOfWeek'] = df.index.dayofweek
            df['Quarter'] = df.index.quarter
            
            logger.info("Time features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding time features: {str(e)}")
            raise
    
    def add_lagged_features(self, df):
        """지연 특성 추가"""
        try:
            logger.info("Adding lagged features")
            
            # 종가에 대한 지연 특성
            for i in [1, 2, 3, 5, 10, 20]:
                df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            
            # 거래량에 대한 지연 특성
            for i in [1, 5, 10]:
                df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
            
            # 수익률 계산
            df['Returns'] = df['Close'].pct_change()
            
            # NaN 값 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info("Lagged features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding lagged features: {str(e)}")
            raise
    
    def add_rolling_features(self, df):
        """이동 평균 특성 추가"""
        try:
            logger.info("Adding rolling features")
            
            # 종가에 대한 이동 평균
            for window in [5, 10, 20, 50]:
                df[f'Close_Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'Close_Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
            
            # 거래량에 대한 이동 평균
            for window in [5, 10, 20]:
                df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
            
            # NaN 값 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info("Rolling features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding rolling features: {str(e)}")
            raise

def main():
    # 예제 사용법
    from data.data_loader import DataLoader
    
    # 데이터 로드
    loader = DataLoader()
    data = loader.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # 특성 엔지니어링
    engineer = FeatureEngineer()
    data = engineer.process(data)
    
    # 특성 및 타겟 준비
    X, y = engineer.prepare_features()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {X.columns.tolist()}")

if __name__ == "__main__":
    main() 