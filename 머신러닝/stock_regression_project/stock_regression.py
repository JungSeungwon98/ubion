import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
import logging
import argparse
from datetime import datetime, timedelta
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockRegressor:
    """주식 가격 예측을 위한 회귀 모델 클래스"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_name = None
        self.results_dir = 'results'
        
    def create_features(self, df):
        """주식 데이터로부터 특성 생성"""
        try:
            # 기본 특성
            df = df.copy()
            
            # 기술적 지표 계산
            # 1. 이동평균
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            
            # 2. RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. 볼린저 밴드
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_std'] = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
            df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
            
            # 4. 거래량 관련 지표
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            
            # 5. 가격 변화율
            df['Returns'] = df['Close'].pct_change()
            df['Returns_MA5'] = df['Returns'].rolling(window=5).mean()
            
            # 6. 변동성
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # NaN 값 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 특성 선택
            feature_columns = [
                'Open', 'High', 'Low', 'Volume',
                'MA5', 'MA20', 'MA60',
                'RSI', 'BB_middle', 'BB_upper', 'BB_lower',
                'Volume_MA5', 'Volume_MA20',
                'Returns', 'Returns_MA5', 'Volatility'
            ]
            
            self.feature_columns = feature_columns
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return None
    
    def prepare_data(self, df, target_column='Close', test_size=0.2):
        """데이터 준비 및 분할"""
        try:
            # 특성 생성
            df = self.create_features(df)
            if df is None:
                return None, None, None, None
            
            # 특성과 타겟 분리
            X = df[self.feature_columns]
            y = df[target_column]
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # 특성 스케일링
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None, None, None, None
    
    def train_model(self, X_train, y_train, model_type='linear', **model_params):
        """모델 학습"""
        try:
            if model_type == 'linear':
                self.model = LinearRegression(**model_params)
                self.model_name = "Linear Regression"
            elif model_type == 'ridge':
                self.model = Ridge(**model_params)
                self.model_name = f"Ridge Regression (alpha={model_params.get('alpha', 1.0)})"
            elif model_type == 'lasso':
                self.model = Lasso(**model_params)
                self.model_name = f"Lasso Regression (alpha={model_params.get('alpha', 1.0)})"
            elif model_type == 'elasticnet':
                self.model = ElasticNet(**model_params)
                self.model_name = f"ElasticNet (alpha={model_params.get('alpha', 1.0)}, l1_ratio={model_params.get('l1_ratio', 0.5)})"
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.model.fit(X_train, y_train)
            logger.info(f"Model trained successfully with {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
    
    def evaluate_model(self, X_test, y_test):
        """모델 평가"""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
            
            # 예측
            y_pred = self.model.predict(X_test)
            
            # 평가 지표 계산
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            explained_var = explained_variance_score(y_test, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # 결과 저장
            results = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2,
                'Explained Variance': explained_var
            }
            
            # 결과 출력
            logger.info(f"Model Evaluation Results for {self.model_name}:")
            logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
            logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
            logger.info(f"Mean Absolute Percentage Error (MAPE): {mae:.2f}%")
            logger.info(f"R-squared (R2): {r2:.4f}")
            logger.info(f"Explained Variance: {explained_var:.4f}")
            
            return y_pred, results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return None, None
    
    def plot_predictions(self, y_test, y_pred, title="Stock Price Prediction", save_plot=True):
        """예측 결과 시각화"""
        try:
            # 결과 디렉토리 생성
            os.makedirs(self.results_dir, exist_ok=True)
            
            # 예측 결과 플롯
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.values, label='Actual', color='blue')
            plt.plot(y_pred, label='Predicted', color='red')
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # 플롯 저장
            if save_plot:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{self.results_dir}/prediction_{timestamp}.png"
                plt.savefig(filename)
                logger.info(f"Prediction plot saved to {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
    
    def plot_residuals(self, y_test, y_pred, save_plot=True):
        """잔차 분석 시각화"""
        try:
            # 결과 디렉토리 생성
            os.makedirs(self.results_dir, exist_ok=True)
            
            # 잔차 계산
            residuals = y_test - y_pred
            
            # 잔차 플롯
            plt.figure(figsize=(12, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title(f"Residual Plot for {self.model_name}")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.grid(True)
            
            # 플롯 저장
            if save_plot:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{self.results_dir}/residuals_{timestamp}.png"
                plt.savefig(filename)
                logger.info(f"Residuals plot saved to {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting residuals: {str(e)}")
    
    def plot_feature_importance(self, save_plot=True):
        """특성 중요도 시각화"""
        try:
            if self.model is None or self.feature_columns is None:
                logger.warning("Model or feature columns not available for feature importance plot")
                return
                
            # 결과 디렉토리 생성
            os.makedirs(self.results_dir, exist_ok=True)
            
            # 특성 중요도 계산 (선형 모델의 경우 계수 절대값 사용)
            if hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_)
                
                # 특성 중요도 플롯
                plt.figure(figsize=(12, 8))
                feature_importance = pd.DataFrame({
                    'Feature': self.feature_columns,
                    'Importance': importance
                })
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                sns.barplot(x='Importance', y='Feature', data=feature_importance)
                plt.title(f"Feature Importance for {self.model_name}")
                plt.xlabel("Absolute Coefficient Value")
                plt.ylabel("Feature")
                plt.tight_layout()
                
                # 플롯 저장
                if save_plot:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{self.results_dir}/feature_importance_{timestamp}.png"
                    plt.savefig(filename)
                    logger.info(f"Feature importance plot saved to {filename}")
                
                plt.show()
            else:
                logger.warning("Model does not have coefficients for feature importance plot")
                
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
    
    def save_results(self, results, symbol):
        """평가 결과를 CSV 파일로 저장"""
        try:
            # 결과 디렉토리 생성
            os.makedirs(self.results_dir, exist_ok=True)
            
            # 결과를 데이터프레임으로 변환
            results_df = pd.DataFrame([results])
            results_df['Model'] = self.model_name
            results_df['Symbol'] = symbol
            results_df['Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 파일 이름 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.results_dir}/results_{symbol}_{timestamp}.csv"
            
            # CSV로 저장
            results_df.to_csv(filename, index=False)
            logger.info(f"Results saved to {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return None

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='Stock Price Prediction using Regression Models')
    
    # 필수 인자
    parser.add_argument('--symbol', type=str, required=True,
                      help='Stock symbol (e.g., AAPL, MSFT, GOOGL)')
    
    # 선택적 인자
    parser.add_argument('--start-date', type=str,
                      help='Start date (YYYY-MM-DD). Default: 1 year ago')
    parser.add_argument('--end-date', type=str,
                      help='End date (YYYY-MM-DD). Default: today')
    parser.add_argument('--model', type=str, default='linear',
                      choices=['linear', 'ridge', 'lasso', 'elasticnet'],
                      help='Regression model type')
    parser.add_argument('--alpha', type=float, default=1.0,
                      help='Regularization strength for Ridge, Lasso, and ElasticNet')
    parser.add_argument('--l1-ratio', type=float, default=0.5,
                      help='L1 ratio for ElasticNet (0 <= l1_ratio <= 1)')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Test set size (0 < test_size < 1)')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save plots to results directory')
    
    return parser.parse_args()

def main():
    # 명령줄 인자 파싱
    args = parse_args()
    
    # 날짜 설정
    if not args.start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
        
    if not args.end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    # 데이터 로더 초기화
    loader = DataLoader()
    
    # 데이터 가져오기
    logger.info(f"Fetching data for {args.symbol} from {start_date} to {end_date}")
    data = loader.fetch_stock_data(args.symbol, start_date, end_date)
    if data is None or data.empty:
        logger.error("Failed to fetch data")
        return
    
    # 회귀 모델 초기화
    regressor = StockRegressor()
    
    # 데이터 준비
    X_train, X_test, y_train, y_test = regressor.prepare_data(
        data, test_size=args.test_size
    )
    if X_train is None:
        logger.error("Failed to prepare data")
        return
    
    # 모델 파라미터 설정
    model_params = {}
    if args.model in ['ridge', 'lasso', 'elasticnet']:
        model_params['alpha'] = args.alpha
    if args.model == 'elasticnet':
        model_params['l1_ratio'] = args.l1_ratio
    
    # 모델 학습
    regressor.train_model(X_train, y_train, model_type=args.model, **model_params)
    
    # 모델 평가
    y_pred, results = regressor.evaluate_model(X_test, y_test)
    if y_pred is None:
        logger.error("Failed to evaluate model")
        return
    
    # 결과 시각화
    title = f"{args.symbol} Stock Price Prediction using {regressor.model_name}"
    regressor.plot_predictions(y_test, y_pred, title=title, save_plot=args.save_plots)
    
    # 잔차 분석
    regressor.plot_residuals(y_test, y_pred, save_plot=args.save_plots)
    
    # 특성 중요도
    regressor.plot_feature_importance(save_plot=args.save_plots)
    
    # 결과 저장
    regressor.save_results(results, args.symbol)
    
    logger.info("Analysis completed successfully!")

if __name__ == "__main__":
    main() 