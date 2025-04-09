import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os

class ModelEvaluator:
    def __init__(self):
        """모델 평가 클래스 초기화"""
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """성능 지표 계산"""
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }
        
        return metrics
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                initial_capital: float = 10000.0) -> Dict[str, float]:
        """트레이딩 성과 지표 계산"""
        # 수익률 계산
        actual_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # 매매 신호 생성 (예측이 실제보다 높으면 매수, 낮으면 매도)
        signals = np.sign(pred_returns)
        
        # 전략 수익률 계산
        strategy_returns = signals[:-1] * actual_returns[1:]
        
        # 누적 수익률
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        
        # 샤프 비율 계산 (연간화)
        risk_free_rate = 0.02  # 2% 무위험 수익률 가정
        excess_returns = strategy_returns - risk_free_rate / 252  # 일간 무위험 수익률
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(strategy_returns)
        
        # 최대 낙폭 계산
        cumulative_peaks = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - cumulative_peaks
        max_drawdown = np.min(drawdowns)
        
        # 승률 계산
        winning_trades = np.sum(strategy_returns > 0)
        total_trades = len(strategy_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 수익성 지표
        final_portfolio_value = initial_capital * (1 + cumulative_returns[-1])
        total_return = (final_portfolio_value - initial_capital) / initial_capital
        
        metrics = {
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Total_Return': total_return,
            'Final_Portfolio_Value': final_portfolio_value,
            'Total_Trades': total_trades
        }
        
        return metrics
    
    def compare_models(self, models_predictions: Dict[str, np.ndarray],
                      y_true: np.ndarray) -> pd.DataFrame:
        """여러 모델의 성능 비교"""
        results = []
        
        for model_name, y_pred in models_predictions.items():
            # 예측 성능 지표
            pred_metrics = self.calculate_metrics(y_true, y_pred)
            
            # 트레이딩 성과 지표
            trading_metrics = self.calculate_trading_metrics(y_true, y_pred)
            
            # 결과 통합
            metrics = {**pred_metrics, **trading_metrics}
            metrics['Model'] = model_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def evaluate_models(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        여러 모델의 예측 결과를 평가하고 성능 지표를 반환합니다.
        
        Parameters:
        -----------
        y_true : np.ndarray
            실제 값 배열
        predictions : Dict[str, np.ndarray]
            모델별 예측 값 딕셔너리 (모델명: 예측값)
        
        Returns:
        --------
        Dict[str, Dict[str, float]]
            모델별 성능 지표 딕셔너리
        """
        self.logger.info("Evaluating model performance...")
        evaluation_results = {}
        
        for model_name, y_pred in predictions.items():
            try:
                # 길이가 다른 경우 조정
                if len(y_pred) != len(y_true):
                    min_len = min(len(y_pred), len(y_true))
                    y_pred_adj = y_pred[:min_len]
                    y_true_adj = y_true[:min_len]
                    self.logger.warning(f"Length mismatch between predictions ({len(y_pred)}) and "
                                       f"true values ({len(y_true)}). Truncating to {min_len}.")
                else:
                    y_pred_adj = y_pred
                    y_true_adj = y_true
                
                # 성능 지표 계산
                metrics = self.calculate_metrics(y_true_adj, y_pred_adj)
                evaluation_results[model_name] = metrics
                self.logger.info(f"Model {model_name} evaluated successfully.")
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {str(e)}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    def plot_predictions(self, dates, y_true, y_pred, model_name='Model', save_path=None):
        """예측 결과 시각화

        단일 모델의 예측 결과를 시각화합니다.
        
        Parameters
        ----------
        dates : DatetimeIndex or array-like
            날짜 인덱스
        y_true : array-like
            실제 값
        y_pred : array-like
            예측 값
        model_name : str, optional
            모델 이름, by default 'Model'
        save_path : str, optional
            저장 경로, by default None
        
        Returns
        -------
        str or None
            저장된 이미지 경로 또는 None
        """
        try:
            # 입력값 확인
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # 길이가 다른 경우 조정
            min_len = min(len(y_true), len(y_pred), len(dates))
            dates = dates[:min_len]
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            plt.figure(figsize=(15, 8))
            plt.plot(dates, y_true, label='Actual', color='black', alpha=0.7)
            plt.plot(dates, y_pred, label=f'{model_name} Prediction', color='blue', alpha=0.5)
            
            plt.title(f'Stock Price Prediction - {model_name}')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Prediction plot saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.close()
                return None
        except Exception as e:
            self.logger.error(f"Error plotting predictions: {str(e)}")
            return None
    
    def plot_multiple_predictions(self, models_predictions: Dict[str, np.ndarray],
                        y_true: np.ndarray, dates: pd.DatetimeIndex,
                        save_dir: str = None):
        """여러 모델의 예측 결과 시각화"""
        try:
            plt.figure(figsize=(15, 8))
            plt.plot(dates, y_true, label='Actual', color='black', alpha=0.7)
            
            colors = plt.cm.rainbow(np.linspace(0, 1, len(models_predictions)))
            for (model_name, y_pred), color in zip(models_predictions.items(), colors):
                plt.plot(dates, y_pred, label=f'{model_name} Prediction',
                        color=color, alpha=0.5)
            
            plt.title('Stock Price Predictions Comparison')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'predictions_comparison.png')
                plt.savefig(save_path)
                self.logger.info(f"Multiple predictions plot saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.close()
                return None
        except Exception as e:
            self.logger.error(f"Error plotting multiple predictions: {str(e)}")
            return None
    
    def plot_returns_distribution(self, models_predictions: Dict[str, np.ndarray],
                                y_true: np.ndarray, save_dir: str = None):
        """수익률 분포 시각화"""
        returns_data = []
        
        # 실제 수익률
        actual_returns = np.diff(y_true) / y_true[:-1]
        returns_data.append(pd.Series(actual_returns, name='Actual'))
        
        # 예측 수익률
        for model_name, y_pred in models_predictions.items():
            pred_returns = np.diff(y_pred) / y_pred[:-1]
            returns_data.append(pd.Series(pred_returns, name=model_name))
        
        returns_df = pd.concat(returns_data, axis=1)
        
        plt.figure(figsize=(15, 8))
        sns.kdeplot(data=returns_df, fill=True)
        plt.title('Returns Distribution Comparison')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.grid(True)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'returns_distribution.png'))
        plt.show()
    
    def plot_cumulative_returns(self, models_predictions: Dict[str, np.ndarray],
                              y_true: np.ndarray, dates: pd.DatetimeIndex,
                              save_dir: str = None):
        """누적 수익률 시각화"""
        plt.figure(figsize=(15, 8))
        
        # 실제 누적 수익률
        actual_returns = np.diff(y_true) / y_true[:-1]
        cumulative_actual = np.cumprod(1 + actual_returns) - 1
        plt.plot(dates[1:], cumulative_actual, label='Actual',
                color='black', alpha=0.7)
        
        # 예측 기반 전략의 누적 수익률
        colors = plt.cm.rainbow(np.linspace(0, 1, len(models_predictions)))
        for (model_name, y_pred), color in zip(models_predictions.items(), colors):
            pred_returns = np.diff(y_pred) / y_pred[:-1]
            signals = np.sign(pred_returns)
            strategy_returns = signals[:-1] * actual_returns[1:]
            cumulative_strategy = np.cumprod(1 + strategy_returns) - 1
            plt.plot(dates[2:], cumulative_strategy,
                    label=f'{model_name} Strategy', color=color, alpha=0.5)
        
        plt.title('Cumulative Returns Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'cumulative_returns.png'))
        plt.show()
    
    def generate_report(self, y_true, predictions, save_path=None):
        """
        종합 성과 보고서 생성
        
        Parameters
        ----------
        y_true : np.ndarray
            실제 값
        predictions : Dict[str, np.ndarray]
            모델별 예측 값 딕셔너리 (모델명: 예측값)
        save_path : str, optional
            저장 경로, by default None
            
        Returns
        -------
        str
            저장된 보고서 경로
        """
        try:
            # 결과 디렉토리 생성
            save_dir = os.path.dirname(save_path) if save_path else 'reports'
            os.makedirs(save_dir, exist_ok=True)
            
            # 파일 경로 설정
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(save_dir, f'model_evaluation_report_{timestamp}.html')
            
            # 성능 비교 테이블
            evaluation_results = self.evaluate_models(y_true, predictions)
            
            # 결과를 DataFrame으로 변환
            results = []
            for model_name, metrics in evaluation_results.items():
                metrics_copy = metrics.copy()
                metrics_copy['Model'] = model_name
                results.append(metrics_copy)
            
            comparison_df = pd.DataFrame(results)
            
            # HTML 보고서 생성
            with open(save_path, 'w') as f:
                f.write('<html><head>')
                f.write('<style>')
                f.write('body { font-family: Arial, sans-serif; margin: 20px; }')
                f.write('table { border-collapse: collapse; width: 100%; }')
                f.write('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
                f.write('th { background-color: #f2f2f2; }')
                f.write('</style>')
                f.write('</head><body>')
                
                # 제목
                f.write('<h1>Stock Price Prediction Model Evaluation Report</h1>')
                f.write(f'<p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
                
                # 성과 비교 테이블
                f.write('<h2>Model Performance Comparison</h2>')
                f.write(comparison_df.to_html())
                
                f.write('</body></html>')
            
            self.logger.info(f"Report generated and saved to {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None

def main():
    # 예제 사용법
    import yfinance as yf
    from ..data.feature_engineering import FeatureEngineer
    from .linear_models import StockLinearModels
    from .tree_models import StockTreeModels
    from .neural_models import StockNeuralModels
    
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
    dates = X.index
    
    # 각 모델 학습
    linear_models = StockLinearModels()
    tree_models = StockTreeModels()
    neural_models = StockNeuralModels()
    
    linear_results = linear_models.train_models(X, y)
    tree_results = tree_models.train_models(X, y)
    neural_results = neural_models.train_models(X, y)
    
    # 예측 수행
    predictions = {
        'Linear': linear_models.predict(X, 'linear'),
        'Ridge': linear_models.predict(X, 'ridge'),
        'Random_Forest': tree_models.predict(X, 'random_forest'),
        'XGBoost': tree_models.predict(X, 'xgboost'),
        'LSTM': neural_models.predict(X, 'lstm')
    }
    
    # 모델 평가
    evaluator = ModelEvaluator()
    report_path = evaluator.generate_report(y.values, predictions)
    print(f"Evaluation report saved to: {report_path}")

if __name__ == "__main__":
    main() 