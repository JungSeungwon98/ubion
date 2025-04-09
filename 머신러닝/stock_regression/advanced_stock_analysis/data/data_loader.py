import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)

class DataLoader:
    """주식 데이터를 가져오는 클래스"""
    
    def fetch_stock_data(self, symbol: str, start_date=None, end_date=None):
        """yfinance를 사용하여 주식 데이터 가져오기"""
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}...")
            
            # 기본 날짜 설정
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # yfinance에서 데이터 가져오기 (auto_adjust=True로 수정주가 사용)
            stock_data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                progress=True,
                auto_adjust=True,  # 수정주가 사용
                actions=False
            )
            
            # 데이터가 없는 경우
            if stock_data.empty:
                logger.error(f"No data available for {symbol}. Please check the symbol or date range.")
                return pd.DataFrame()
            
            # 데이터 확인
            logger.info(f"Data columns: {stock_data.columns.tolist()}")
            logger.info(f"Data shape: {stock_data.shape}")
            logger.info(f"Data column index type: {type(stock_data.columns)}")
            
            # MultiIndex 처리 (yfinance의 최신 버전은 단일 레벨 컬럼을 반환)
            if isinstance(stock_data.columns, pd.MultiIndex):
                logger.info("MultiIndex columns detected, flattening...")
                
                # 새로운 컬럼 목록 생성
                new_cols = []
                for col in stock_data.columns:
                    if isinstance(col, tuple) and len(col) > 1:
                        if col[1] == symbol:  # 심볼 부분 제거
                            new_cols.append(col[0])
                        else:
                            new_cols.append("_".join([str(c) for c in col if c]))
                    else:
                        new_cols.append(col)
                
                # 새 데이터프레임 생성
                df = pd.DataFrame(stock_data.values, index=stock_data.index, columns=new_cols)
            else:
                # MultiIndex가 아닌 경우
                df = stock_data.copy()
            
            # 열 이름 대문자로 통일
            df.columns = [col.title() if isinstance(col, str) else col for col in df.columns]
            
            # 'Adj Close' 열이 있으면 제거하고 'Close'만 사용
            if 'Adj Close' in df.columns and 'Close' in df.columns:
                df = df.drop(columns=['Adj Close'])
            elif 'Adj Close' in df.columns and 'Close' not in df.columns:
                df = df.rename(columns={'Adj Close': 'Close'})
            
            # 필수 열 있는지 확인
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Column {col} not found in data")
            
            # NaN 값 개수 확인
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Data contains {nan_count} missing values")
                # NaN 값 채우기 (forward fill 후 backward fill)
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 샘플 데이터 로깅
            logger.info("Data sample:")
            with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
                logger.info(df.head(3).to_string())
            
            logger.info(f"Successfully fetched {len(df)} rows of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()  # 빈 데이터프레임 반환
    
    def save_data(self, df, filename=None, directory='data'):
        """데이터를 CSV 파일로 저장"""
        try:
            if df is None or df.empty:
                logger.error("No data to save")
                return None
            
            # 디렉토리 생성
            os.makedirs(directory, exist_ok=True)
            
            # 파일 이름 생성
            if filename is None:
                symbol = df.index.name if df.index.name else 'stock'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{symbol}_{timestamp}.csv"
            
            # 경로 결합
            filepath = os.path.join(directory, filename)
            
            # CSV로 저장
            df.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None
    
    def load_data(self, filepath):
        """CSV 파일에서 데이터 로드"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None
            
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded data from {filepath}, shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            return None

def main():
    # 예제 사용법
    loader = DataLoader()
    data = loader.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
    print(data.head())

if __name__ == "__main__":
    main() 