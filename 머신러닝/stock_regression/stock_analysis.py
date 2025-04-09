import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

class StockAnalyzer:
    def __init__(self, symbol, start_date, end_date):
        """
        주식 분석기 초기화
        
        Parameters:
        -----------
        symbol : str
            주식 심볼 (예: 'AAPL' for Apple)
        start_date : str
            시작 날짜 (YYYY-MM-DD)
        end_date : str
            종료 날짜 (YYYY-MM-DD)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_data(self):
        """주식 데이터 다운로드 및 전처리"""
        # Yahoo Finance에서 데이터 다운로드
        stock = yf.Ticker(self.symbol)
        self.data = stock.history(start=self.start_date, end=self.end_date)
        
        # 기술적 지표 계산
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        
        # 결측값 제거
        self.data = self.data.dropna()
        
        return self.data
    
    def prepare_features(self):
        """특성 준비 및 전처리"""
        # 특성 선택
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'Volatility']
        X = self.data[features]
        y = self.data['Close']
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """선형 회귀 모델 학습"""
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """모델 성능 평가"""
        y_pred = self.model.predict(X_test)
        
        # 성능 지표 계산
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared Score: {r2:.2f}")
        
        return mse, r2
    
    def plot_results(self, X_test, y_test):
        """결과 시각화"""
        y_pred = self.model.predict(X_test)
        
        # 실제값 vs 예측값 산점도
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Stock Prices')
        plt.tight_layout()
        plt.show()
        
        # 특성 중요도 시각화
        feature_importance = pd.DataFrame({
            'Feature': ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'Volatility'],
            'Importance': np.abs(self.model.coef_)
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance in Linear Regression Model')
        plt.tight_layout()
        plt.show()

def main():
    # 분석 파라미터 설정
    symbol = 'AAPL'  # Apple 주식
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    # 분석기 초기화 및 실행
    analyzer = StockAnalyzer(symbol, start_date, end_date)
    data = analyzer.fetch_data()
    print(f"\n데이터 샘플:\n{data.head()}")
    
    # 특성 준비
    X_train, X_test, y_train, y_test = analyzer.prepare_features()
    
    # 모델 학습
    model = analyzer.train_model(X_train, y_train)
    print("\n모델 계수:")
    for feature, coef in zip(['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'Volatility'], 
                           model.coef_):
        print(f"{feature}: {coef:.4f}")
    
    # 모델 평가
    print("\n모델 성능 평가:")
    analyzer.evaluate_model(X_test, y_test)
    
    # 결과 시각화
    analyzer.plot_results(X_test, y_test)

if __name__ == "__main__":
    main() 