import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class MarketDataLoader:
    def __init__(self):
        self.data_dir = "sample_data"
        
        # Create sample data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_market_data(self, symbol, days=365):
        """Load market data for a given symbol and VIX index"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            # Generate data
            print(f"Generating data for {symbol}...")
            stock_data = self._generate_stock_data(symbol, start_date, end_date)
            
            # Generate VIX data
            print("Generating VIX data...")
            vix_data = self._generate_vix_data(start_date, end_date)
            
            # Merge stock and VIX data
            data = pd.DataFrame()
            data['Close'] = stock_data['Close']
            data['Volume'] = stock_data['Volume']
            data['VIX'] = vix_data['Close']
            
            # Calculate daily returns
            data['Returns'] = data['Close'].pct_change()
            
            # Calculate volatility (20-day rolling standard deviation of returns)
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            
            # Drop rows with NaN values
            data = data.dropna()
            
            if data.empty:
                raise ValueError("No valid data available after processing")
            
            print(f"Successfully generated {len(data)} days of data")
            return data
            
        except Exception as e:
            print(f"Error generating market data: {str(e)}")
            raise

    def _generate_stock_data(self, symbol, start_date, end_date):
        """Generate synthetic stock data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame(index=dates)
        
        # Base value depends on symbol
        if symbol == "AAPL":
            base_value = 150
        elif symbol == "MSFT":
            base_value = 250
        elif symbol == "GOOGL":
            base_value = 2000
        elif symbol == "AMZN":
            base_value = 3000
        else:
            base_value = 100
        
        # Generate price data
        data['Open'] = np.random.normal(base_value, base_value * 0.05, len(dates))
        data['High'] = data['Open'] + np.random.uniform(0, base_value * 0.02, len(dates))
        data['Low'] = data['Open'] - np.random.uniform(0, base_value * 0.02, len(dates))
        data['Close'] = np.random.normal(data['Open'], base_value * 0.01, len(dates))
        data['Volume'] = np.random.randint(1000000, 10000000, len(dates))
        
        # Add trend and seasonality
        trend = np.linspace(0, base_value * 0.2, len(dates))
        seasonality = base_value * 0.05 * np.sin(np.arange(len(dates)) * 0.1)
        data['Close'] = data['Close'] + trend + seasonality
        
        # Ensure High is highest and Low is lowest
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        
        return data

    def _generate_vix_data(self, start_date, end_date):
        """Generate synthetic VIX data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame(index=dates)
        
        # Base VIX value
        base_value = 20
        
        # Generate VIX data
        data['Open'] = np.random.normal(base_value, 5, len(dates))
        data['High'] = data['Open'] + np.random.uniform(0, 3, len(dates))
        data['Low'] = data['Open'] - np.random.uniform(0, 3, len(dates))
        data['Close'] = np.random.normal(data['Open'], 1, len(dates))
        data['Volume'] = np.random.randint(100000, 1000000, len(dates))
        
        # Add trend and seasonality
        trend = np.linspace(0, 10, len(dates))
        seasonality = 5 * np.sin(np.arange(len(dates)) * 0.05)
        data['Close'] = data['Close'] + trend + seasonality
        
        # Ensure High is highest and Low is lowest
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        
        return data

    def prepare_features(self, data):
        """Prepare features for the model"""
        features = pd.DataFrame()
        
        # Price-based features
        features['Returns'] = data['Returns']
        features['Volatility'] = data['Volatility']
        
        # Volume-based features
        features['Volume_Change'] = data['Volume'].pct_change()
        features['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # VIX-based features
        features['VIX_Change'] = data['VIX'].pct_change()
        features['VIX_MA'] = data['VIX'].rolling(window=20).mean()
        
        # Create target variable (1 if volatility increases, 0 otherwise)
        features['target'] = (data['Volatility'].shift(-1) > data['Volatility']).astype(int)
        print( features['target'].value_counts())
        # Drop rows with NaN values
        features = features.dropna()
        
        if features.empty:
            raise ValueError("No valid features available after processing")
        
        return features

if __name__ == "__main__":
    # 테스트 코드
    loader = MarketDataLoader()
    features = loader.prepare_features(loader.load_market_data("AAPL"))
    print("Features shape:", features.shape)
    print("\nFeatures head:")
    print(features.head()) 
# if __name__ == "__main__":
#     # 테스트 코드
#     loader = MarketDataLoader()
#     market_data = loader.load_market_data("AAPL")
    
#     # 시작 날짜와 종료 날짜를 설정
#     start_date = market_data.index.min().strftime('%Y-%m-%d')
#     end_date = market_data.index.max().strftime('%Y-%m-%d')
    
#     # prepare_features 호출 시 start_date와 end_date 전달
#     features = loader.prepare_features(market_data, start_date, end_date)
#     print("Features shape:", features.shape)
#     print("\nFeatures head:")
#     print(features.head())