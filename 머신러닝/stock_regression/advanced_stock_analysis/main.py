import os
import argparse
import logging
import sys
import json
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

# GUI 백엔드 문제 방지를 위한 matplotlib 설정
import matplotlib
matplotlib.use('Agg')

# 프로젝트 모듈 임포트
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.linear_models import LinearModel
from models.tree_models import TreeModel
from models.neural_models import NeuralModel
from models.model_evaluation import ModelEvaluator
# from visualization.visualization import StockVisualizer  # 시각화 모듈 비활성화
from app import create_app
from dashboard import StockDashboard

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='주식 분석 도구')
    
    # 모드 선택
    parser.add_argument('--mode', type=str, choices=['cli', 'web', 'dashboard'], default='cli',
                        help='실행 모드 (cli, web, dashboard)')
    
    # 주식 심볼 및 날짜 범위
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='주식 심볼 (예: AAPL)')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-01',
                       help='종료 날짜 (YYYY-MM-DD)')
    
    # 모델 선택
    parser.add_argument('--model', type=str, choices=['linear', 'tree', 'neural'], default='tree',
                       help='사용할 모델 (linear, tree, neural)')
    
    # 웹 서버 설정
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='웹 서버 호스트 (기본값: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8050,
                       help='웹 서버 포트 (기본값: 8050)')
    
    # CSV 파일 경로
    parser.add_argument('--csv', type=str, default=None,
                       help='CSV 파일 경로 (데이터를 API 대신 CSV에서 로드)')
    
    return parser.parse_args()

def run_cli_mode(args):
    """CLI 모드 실행"""
    logger.info(f"Running in CLI mode for symbol: {args.symbol}")
    
    # 데이터 로더 초기화
    data_loader = DataLoader()
    
    # 데이터 가져오기
    df = data_loader.fetch_stock_data(args.symbol, args.start_date, args.end_date)
    
    # 데이터가 비어있는지 확인
    if df.empty:
        logger.error(f"No data available for {args.symbol}. Please check the symbol or date range.")
        return None
    
    # 멀티인덱스 컬럼을 단일 문자열로 변환
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # 컬럼 이름 출력
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    
    logger.info(f"Fetched {len(df)} rows of data for {args.symbol}")
    
    # 특성 엔지니어링
    try:
        engineer = FeatureEngineer()
        
        # 선형 모델만 사용하는 경우 기술적 지표 계산을 건너뜀
        if args.model == 'linear':
            df = engineer.add_time_features(df)
            df = engineer.add_lagged_features(df)
            df = engineer.add_rolling_features(df)
            logger.info(f"Added {len(df.columns) - 6} features to the dataset (skipped technical indicators for linear model)")
        else:
            df = engineer.add_technical_indicators(df)
            df = engineer.add_time_features(df)
            df = engineer.add_lagged_features(df)
            df = engineer.add_rolling_features(df)
            logger.info(f"Added {len(df.columns) - 6} features to the dataset")
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        return None
    
    # NaN 값 처리
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 데이터 분할
    if len(df) < 30:  # 최소 데이터 포인트 확인
        logger.error(f"Insufficient data points ({len(df)}) for {args.symbol}. Need at least 30 data points.")
        return None
    
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    logger.info(f"Split data into {len(train_data)} training and {len(test_data)} test samples")
    
    # 특성 및 타겟 설정
    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
    target_col = 'Returns'
    
    # 특성 및 타겟 분리
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    # NaN 값 확인
    if X_train.isna().any().any() or y_train.isna().any():
        logger.warning("NaN values found in training data. Filling with forward and backward fill.")
        X_train = X_train.fillna(method='ffill').fillna(method='bfill')
        y_train = y_train.fillna(method='ffill').fillna(method='bfill')
    
    if X_test.isna().any().any() or y_test.isna().any():
        logger.warning("NaN values found in test data. Filling with forward and backward fill.")
        X_test = X_test.fillna(method='ffill').fillna(method='bfill')
        y_test = y_test.fillna(method='ffill').fillna(method='bfill')
    
    # 모델 학습 및 평가
    models = {}
    evaluator = ModelEvaluator()
    
    # 선형 모델
    if args.model in ['linear', 'all']:
        try:
            logger.info(f"Training linear model (type: {args.linear_type})...")
            linear_model = LinearModel(model_type=args.linear_type)
            linear_results = linear_model.train(X_train, y_train)
            models['linear'] = linear_model
            logger.info(f"Linear model ({args.linear_type}) trained with MSE: {linear_results['cv_scores']['mse']['mean']:.6f}")
        except Exception as e:
            logger.error(f"Error training linear model: {str(e)}")
    
    # 트리 모델
    if args.model in ['tree', 'all']:
        try:
            logger.info("Training tree model...")
            tree_model = TreeModel(model_type='random_forest')
            tree_results = tree_model.train(X_train, y_train)
            models['tree'] = tree_model
            logger.info(f"Tree model trained with MSE: {tree_results['cv_scores']['mse']['mean']:.6f}")
        except Exception as e:
            logger.error(f"Error training tree model: {str(e)}")
    
    # 신경망 모델
    if args.model in ['neural', 'all']:
        try:
            logger.info("Training neural network model...")
            neural_model = NeuralModel()
            neural_results = neural_model.train_models(X_train, y_train)
            models['neural'] = neural_model
            logger.info(f"Neural network model trained with MSE: {neural_results['lstm']['cv_scores']['mse'][0]:.6f}")
        except Exception as e:
            logger.error(f"Error training neural model: {str(e)}")
            # 오류가 발생해도 계속 진행
            pass
    
    # 모델 평가
    if not models:
        logger.error("No models were successfully trained.")
        return None
    
    # 모델 평가 및 시각화
    try:
        for name, model in models.items():
            try:
                # 예측
                if name == 'neural':
                    # 신경망 모델은 시퀀스 길이에 따라 샘플 수가 달라질 수 있음
                    y_pred = model.predict(X_test, model_name='lstm')
                    # 예측 결과와 실제 데이터의 길이가 다른 경우 처리
                    if len(y_pred) != len(y_test):
                        logger.warning(f"Prediction length ({len(y_pred)}) does not match test data length ({len(y_test)}). Adjusting...")
                        # 더 짧은 길이에 맞춰 조정
                        min_len = min(len(y_pred), len(y_test))
                        y_pred = y_pred[:min_len]
                        y_test_adjusted = y_test[:min_len]
                    else:
                        y_test_adjusted = y_test
                else:
                    y_pred = model.predict(X_test)
                    y_test_adjusted = y_test
                
                # 평가
                metrics = evaluator.calculate_metrics(y_test_adjusted, y_pred)
                logger.info(f"{name.capitalize()} model metrics: {metrics}")
                
                # 특성 중요도 (가능한 경우)
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance(feature_cols)
                    logger.info(f"{name.capitalize()} model feature importance: {importance.head()}")
            except Exception as e:
                logger.error(f"Error evaluating {name} model: {str(e)}")
                continue
        
        # 시각화
        # visualizer = StockVisualizer()
        # visualizer.create_dashboard(df)
        logger.info("Dashboard creation skipped.")
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
    
    return models

def run_web_mode(args):
    """웹 모드 실행"""
    logger.info(f"Running in web mode on {args.host}:{args.port}")
    
    # Flask 앱 생성 및 실행
    app = create_app()
    app.run(debug=True, host=args.host, port=args.port)

def run_dashboard_mode(args):
    """대시보드 모드 실행"""
    try:
        logging.info(f"Running in dashboard mode on {args.host}:{args.port}")
        
        # 대시보드 폴더 확인
        dashboard_dir = os.path.join(os.getcwd(), 'dashboard')
        if not os.path.exists(dashboard_dir):
            os.makedirs(dashboard_dir)
            logging.info(f"Created dashboard directory at {dashboard_dir}")
        else:
            logging.info(f"Using existing dashboard directory at {dashboard_dir}")
            # 디렉토리 내용 확인
            files = os.listdir(dashboard_dir)
            logging.info(f"Dashboard directory contents: {files}")
        
        # 대시보드 실행
        from dashboard import StockDashboard
        dashboard = StockDashboard()
        
        # 정적 파일 경로 확인
        logging.info(f"Static folder: {dashboard.app.server.static_folder}")
        logging.info(f"Static URL path: {dashboard.app.server.static_url_path}")
        
        # 주요 경로 출력
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"App instance path: {dashboard.app.server.instance_path if hasattr(dashboard.app.server, 'instance_path') else 'N/A'}")
        
        # 대시보드 실행
        dashboard.run(debug=True, host=args.host, port=args.port)
        
    except Exception as e:
        logging.error(f"Error in dashboard mode: {str(e)}")
        traceback.print_exc()

def train_and_evaluate_model(data, model_type='tree', target_column='Close', cv=5):
    try:
        logging.info(f"Training and evaluating {model_type} model...")
        
        # 데이터가 비어있는지 확인
        if data is None or data.empty:
            logging.error("No data available for training")
            return None, None, None
        
        # 데이터 형태 확인
        logging.info(f"Data shape: {data.shape}")
        logging.info(f"Data columns: {data.columns.tolist()}")
        
        # NaN 값 검사
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            logging.warning(f"Data contains {nan_count} missing values")
            # NaN 값 처리
            data = data.fillna(method='ffill').fillna(method='bfill')
            # 여전히 NaN 값이 있는지 확인
            nan_count = data.isna().sum().sum()
            if nan_count > 0:
                logging.warning(f"After ffill/bfill, still have {nan_count} missing values. Using SimpleImputer.")
                from sklearn.impute import SimpleImputer
                # 'Date' 열이 있다면 제외하고 처리
                cols_to_impute = [col for col in data.columns if col != 'Date']
                imputer = SimpleImputer(strategy='mean')
                data[cols_to_impute] = imputer.fit_transform(data[cols_to_impute])
        
        # 특성과 타겟 분리
        if target_column not in data.columns:
            logging.error(f"Target column '{target_column}' not found in data")
            return None, None, None
        
        # 레이블 인코딩
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in data.select_dtypes(include=['object']).columns:
            logging.info(f"Label encoding column: {col}")
            data[col] = le.fit_transform(data[col])
            
        # 특성과 타겟 분리
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        logging.info(f"Feature matrix shape: {X.shape}")
        logging.info(f"Target vector shape: {y.shape}")
        
        # 데이터 분할
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        
        # 검증 데이터의 NaN 값 확인 및 처리
        train_nan_count = X_train.isna().sum().sum()
        test_nan_count = X_test.isna().sum().sum()
        
        if train_nan_count > 0 or test_nan_count > 0:
            logging.warning(f"NaN values found in training data: {train_nan_count}")
            logging.warning(f"NaN values found in test data: {test_nan_count}")
            
            # 훈련 데이터 NaN 처리
            X_train = X_train.fillna(method='ffill').fillna(method='bfill')
            X_test = X_test.fillna(method='ffill').fillna(method='bfill')
            
            # 여전히 NaN 값이 있는지 확인
            train_nan_count = X_train.isna().sum().sum()
            test_nan_count = X_test.isna().sum().sum()
            
            if train_nan_count > 0 or test_nan_count > 0:
                logging.warning(f"After ffill/bfill, still have NaN values. Using SimpleImputer.")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # 모델 생성
        if model_type == 'tree':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_type == 'neural':
            try:
                from advanced_stock_analysis.models.neural_models import NeuralModel
                model = NeuralModel(input_dim=X_train.shape[1])
            except ImportError:
                logging.error("Neural model import failed. Falling back to LinearRegression.")
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
        else:
            logging.warning(f"Unknown model type: {model_type}. Using RandomForest")
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 모델 학습 전 데이터 검증
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            logging.error("Empty training data. Cannot train model.")
            return None, None, None
        
        # 모델 학습
        try:
            logging.info(f"Training {model_type} model...")
            model.fit(X_train, y_train)
            logging.info("Model training complete")
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            
            # 데이터셋 문제 확인
            logging.info(f"X_train info: rows={X_train.shape[0]}, columns={X_train.shape[1]}")
            logging.info(f"y_train info: length={len(y_train)}")
            
            # 재시도: 다른 모델로 시도
            try:
                logging.info("Falling back to simple LinearRegression")
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)
                logging.info("Fallback model training complete")
            except Exception as e2:
                logging.error(f"Fallback model training also failed: {str(e2)}")
                return None, None, None
        
        # 교차 검증
        try:
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            logging.info(f"Cross-validation scores: {cv_scores}")
            logging.info(f"Average CV score: {cv_scores.mean()}")
        except Exception as e:
            logging.error(f"Error in cross-validation: {str(e)}")
            cv_scores = None
        
        # 테스트 데이터로 예측
        try:
            y_pred = model.predict(X_test)
            
            # 평가 지표 계산
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"Model Evaluation Metrics:")
            logging.info(f"MSE: {mse}")
            logging.info(f"MAE: {mae}")
            logging.info(f"R^2: {r2}")
            
            # 특성 중요도 (트리 기반 모델인 경우)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importances = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                logging.info("Feature importances:")
                logging.info(feature_importances.head(10))
            
            return model, (mse, mae, r2), cv_scores
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            return model, None, cv_scores
            
    except Exception as e:
        logging.error(f"Unexpected error in train_and_evaluate_model: {str(e)}")
        return None, None, None

def main():
    """메인 함수"""
    # 명령행 인수 파싱
    args = parse_args()
    
    # 실행 모드에 따라 다른 함수 호출
    if args.mode == 'cli':
        try:
            logging.info(f"Running in CLI mode for symbol {args.symbol}")
            
            # 데이터 로더 생성
            loader = DataLoader()
            
            # 주식 데이터 가져오기
            df = loader.fetch_stock_data(args.symbol, args.start_date, args.end_date)
            
            # 데이터가 비어있는지 확인
            if df.empty:
                logging.error(f"No data available for {args.symbol}")
                return
                
            logging.info(f"Successfully fetched {len(df)} rows of data")
            
            # 컬럼이 MultiIndex인 경우 처리
            if isinstance(df.columns, pd.MultiIndex):
                logging.info("Flattening MultiIndex columns")
                # 컬럼 이름을 문자열로 변환
                df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
                logging.info(f"Flattened columns: {df.columns.tolist()}")
            
            # 특성 엔지니어링 (모델 유형이 linear가 아닌 경우에만)
            if args.model != 'linear':
                try:
                    from advanced_stock_analysis.features.feature_engineering import FeatureEngineer
                    engineer = FeatureEngineer()
                    
                    # Returns 열 추가
                    if 'Returns' not in df.columns and 'Close' in df.columns:
                        logging.info("Adding Returns column")
                        df['Returns'] = df['Close'].pct_change()
                    
                    # 기술적 지표 추가
                    df = engineer.add_technical_indicators(df)
                    logging.info(f"Added technical indicators. New shape: {df.shape}")
                    
                    # 시간 기반 특성 추가
                    df = engineer.add_time_features(df)
                    logging.info(f"Added time features. New shape: {df.shape}")
                    
                    # 지연 및 롤링 특성 추가 - Close 열이 있어야 함
                    if 'Close' in df.columns:
                        df = engineer.add_lag_features(df, column='Close', lags=[1, 2, 3, 5, 10])
                        logging.info(f"Added lag features. New shape: {df.shape}")
                    
                        df = engineer.add_rolling_features(df, column='Close', windows=[5, 10, 20])
                        logging.info(f"Added rolling features. New shape: {df.shape}")
                    else:
                        logging.warning("'Close' column not found. Skipping lag and rolling features.")
                    
                    # 특성 엔지니어링 후 NaN 값 처리
                    nan_count = df.isna().sum().sum()
                    if nan_count > 0:
                        logging.warning(f"Data contains {nan_count} missing values after feature engineering")
                        # NaN 값을 포함한 행 제거 (주의: 데이터의 상당 부분이 손실될 수 있음)
                        # df = df.dropna()
                        # 또는 NaN 값 채우기
                        df = df.fillna(method='ffill').fillna(method='bfill')
                        # 남은 NaN 값 평균으로 채우기
                        df = df.fillna(df.mean())
                        # 여전히 NaN 값이 있다면 0으로 채우기
                        df = df.fillna(0)
                        logging.info(f"Processed missing values. New shape: {df.shape}")
                        
                except ImportError as e:
                    logging.error(f"Error importing feature engineering module: {str(e)}")
                except Exception as e:
                    logging.error(f"Error in feature engineering: {str(e)}")
            
            # 데이터 분할
            train_size = int(len(df) * 0.8)
            train_data = df.iloc[:train_size]
            test_data = df.iloc[train_size:]
            
            logging.info(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples")
            
            # 모델 학습 및 평가
            model, metrics, cv_scores = train_and_evaluate_model(
                train_data, 
                model_type=args.model, 
                target_column='Close'
            )
            
            if model is None:
                logging.error("No models were successfully trained")
                return
            
            # 예측 및 시각화
            if metrics:
                mse, mae, r2 = metrics
                logging.info(f"Model metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")
                
                # 대시보드 생성 비활성화
                '''
                try:
                    # 상대 경로 수정
                    from visualization.visualization import StockVisualizer
                    visualizer = StockVisualizer(df)
                    dashboard_path = visualizer.create_dashboard(
                        title=f"{args.symbol} Stock Analysis",
                        model=model,
                        test_data=test_data,
                        feature_cols=[col for col in test_data.columns if col != 'Close'],
                        target_col='Close'
                    )
                    logging.info(f"Dashboard created at {dashboard_path}")
                except Exception as e:
                    logging.error(f"Error creating dashboard: {str(e)}")
                '''
                logging.info("Dashboard creation skipped.")
            
        except Exception as e:
            logging.error(f"Error in CLI mode: {str(e)}")
    elif args.mode == 'web':
        run_web_mode(args)
    elif args.mode == 'dashboard':
        run_dashboard_mode(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 