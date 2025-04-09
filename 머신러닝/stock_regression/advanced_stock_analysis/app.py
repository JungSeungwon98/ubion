import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

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
from utils.visualization import StockVisualizer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# 대시보드 데이터 저장 경로
DASHBOARD_DIR = Path('dashboard')
DASHBOARD_DIR.mkdir(exist_ok=True)

# 전역 변수로 데이터와 모델 저장
stock_data = {}
models = {}
evaluator = None
visualizer = None

@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 폼 데이터 가져오기
        symbol = request.form.get('symbol')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        model_type = request.form.get('model')
        linear_type = request.form.get('linear_type', 'linear')  # 기본값: 'linear'

        # 데이터 로딩
        data_loader = DataLoader()
        df = data_loader.fetch_stock_data(symbol, start_date, end_date)
        
        if df is None or df.empty:
            return render_template('index.html', error='데이터를 가져오는데 실패했습니다.')

        # 특성 엔지니어링
        engineer = FeatureEngineer()
        if model_type == 'linear':
            # 선형 모델의 경우 기술적 지표 계산 건너뛰기
            df = engineer.add_time_features(df)
            df = engineer.add_lagged_features(df)
            df = engineer.add_rolling_features(df)
        else:
            df = engineer.process(df)

        # 모델 선택 및 훈련
        if model_type == 'linear':
            model = LinearModel(model_type=linear_type)
        elif model_type == 'tree':
            model = TreeModel()
        else:
            model = NeuralModel()

        # 데이터 분할
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        # 특성 및 타겟 설정
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        X = df[feature_cols]
        y = df['Returns']

        # 모델 훈련 및 평가
        try:
            metrics = model.train(X, y)
            
            # 결과 포맷팅
            if model_type == 'linear' or model_type == 'tree':
                # 선형 모델 유형을 표시
                model_name = model_type
                if model_type == 'linear':
                    model_name = f"{model_type} ({linear_type})"
                
                result = f"""
모델 유형: {model_name}
평가 지표:
- MSE: {metrics['cv_scores']['mse']['mean']:.6f}
- RMSE: {metrics['cv_scores']['rmse']['mean']:.6f}
- MAE: {metrics['cv_scores']['mae']['mean']:.6f}
- R2: {metrics['cv_scores']['r2']['mean']:.6f}
                """
            elif model_type == 'neural':
                # 신경망 모델은 train_models 메서드를 사용하므로 다른 방식으로 처리해야 함
                try:
                    # 가장 성능이 좋은 모델 선택 (LSTM 또는 CNN-LSTM)
                    model_key = list(metrics.keys())[0] if metrics else 'lstm'
                    result = f"""
모델 유형: {model_type} ({model_key})
평가 지표:
- MSE: {metrics[model_key]['cv_scores']['mse'][0]:.6f}
- RMSE: {metrics[model_key]['cv_scores']['rmse'][0]:.6f}
- MAE: {metrics[model_key]['cv_scores']['mae'][0]:.6f}
- R2: {metrics[model_key]['cv_scores']['r2'][0]:.6f}
                    """
                except (KeyError, IndexError, TypeError) as e:
                    # 에러 발생 시 기본 템플릿 사용
                    logger.error(f"Error formatting neural model results: {str(e)}")
                    result = f"""
모델 유형: {model_type}
평가 지표: 계산 중 오류가 발생했습니다.
                    """
            else:
                result = f"""
모델 유형: {model_type}
평가 지표:
- MSE: {metrics.get('MSE', 'N/A')}
- R2: {metrics.get('R2', 'N/A')}
- MAPE: {metrics.get('MAPE', 'N/A')}
                """
                
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            result = f"""
모델 유형: {model_type}
오류: 모델 훈련 중 오류가 발생했습니다.
- {str(e)}
            """
        
        return render_template('index.html', result=result)

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return render_template('index.html', error=f'분석 중 오류가 발생했습니다: {str(e)}')

@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    """주식 데이터 가져오기 API"""
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        start_date = data.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        include_indicators = data.get('include_indicators', True)
        
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        # 데이터 로더를 통한 데이터 가져오기
        data_loader = DataLoader()
        df = data_loader.fetch_stock_data(symbol, start_date, end_date)
        
        if df is None or df.empty:
            logger.warning(f"No data available for {symbol}")
            return jsonify({'success': False, 'error': f'데이터를 가져올 수 없습니다. 심볼({symbol})과 날짜 범위를 확인하세요.'})
        
        # 멀티인덱스 컬럼을 단일 문자열로 변환
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Converting MultiIndex columns to strings")
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
        # 데이터 로깅
        logger.info(f"Data shape after loading: {df.shape}")
        logger.info(f"Data columns: {df.columns.tolist()}")
        
        # 특성 엔지니어링
        try:
            engineer = FeatureEngineer()
            
            # 기술적 지표 포함 여부에 따라 처리
            if include_indicators:
                df = engineer.process(df)
            else:
                # 기술적 지표 없이 기본 특성만 추가
                df = engineer.add_time_features(df)
                df = engineer.add_lagged_features(df)
                df = engineer.add_rolling_features(df)
            
            # NaN 값 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Data shape after feature engineering: {df.shape}")
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            # 특성 엔지니어링 실패 시 원본 데이터 사용
            
        # 전역 변수에 저장
        stock_data[symbol] = df
        
        # 시각화 도구 초기화
        global visualizer
        visualizer = StockVisualizer()
        
        # 대시보드 생성 시도
        try:
            dashboard_path = visualizer.create_dashboard(df, output_dir=str(DASHBOARD_DIR))
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            dashboard_path = None
        
        # 데이터 요약 정보 반환
        summary = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'rows': len(df),
            'columns': list(df.columns),
            'dashboard_path': dashboard_path
        }
        
        return jsonify({'success': True, 'data': summary})
    
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_models', methods=['POST'])
def train_models():
    """모델 학습 API"""
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        model_type = data.get('model_type', 'tree')
        
        # 모델 유형별 파라미터
        linear_type = data.get('linear_type', 'linear')
        tree_type = data.get('tree_type', 'random_forest')
        neural_type = data.get('neural_type', 'lstm')
        
        # 고급 옵션
        cv_folds = int(data.get('cv_folds', 5))
        test_size = float(data.get('test_size', 0.2))
        
        if symbol not in stock_data:
            return jsonify({'success': False, 'error': 'Data not found. Please fetch data first.'})
        
        df = stock_data[symbol]
        
        # 데이터 분할
        train_size = int(len(df) * (1 - test_size))
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        
        # 특성 및 타겟 설정
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        target_col = 'Returns'
        
        # 모델 학습
        global models
        models = {}
        
        # 요청된 모델 유형에 따라 학습
        if model_type == 'linear' or model_type == 'all':
            # 선형 모델
            linear_model = LinearModel(model_type=linear_type, n_splits=cv_folds)
            linear_model.train(train_data[feature_cols], train_data[target_col])
            models['linear'] = linear_model
            logger.info(f"Linear model ({linear_type}) trained successfully")
        
        if model_type == 'tree' or model_type == 'all':
            # 트리 모델
            tree_model = TreeModel(model_type=tree_type, n_splits=cv_folds)
            tree_model.train(train_data[feature_cols], train_data[target_col])
            models['tree'] = tree_model
            logger.info(f"Tree model ({tree_type}) trained successfully")
        
        if model_type == 'neural' or model_type == 'all':
            # 신경망 모델
            neural_model = NeuralModel(n_splits=cv_folds)
            
            # 신경망 모델 유형에 따라 다른 메서드 호출
            if neural_type == 'dense':
                models_config = {
                    'dense': {
                        'model': neural_model.build_dense_model((train_data[feature_cols].shape[1],))
                    }
                }
            elif neural_type == 'lstm':
                models_config = {
                    'lstm': {
                        'model': neural_model.build_lstm_model((neural_model.sequence_length, train_data[feature_cols].shape[1]))
                    }
                }
            elif neural_type == 'cnn_lstm':
                models_config = {
                    'cnn_lstm': {
                        'model': neural_model.build_cnn_lstm_model((neural_model.sequence_length, train_data[feature_cols].shape[1]))
                    }
                }
            else:
                models_config = None  # 기본 모델 사용
                
            neural_model.train(train_data[feature_cols], train_data[target_col])
            models['neural'] = neural_model
            logger.info(f"Neural model ({neural_type}) trained successfully")
        
        # 모델이 하나도 없으면 오류 반환
        if not models:
            return jsonify({'success': False, 'error': 'No models were trained. Please check model type.'})
        
        # 모델 평가
        global evaluator
        evaluator = ModelEvaluator()
        
        # 예측 수행
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(test_data[feature_cols])
        
        # 평가 결과
        evaluation_results = evaluator.evaluate_models(
            test_data[target_col], 
            predictions
        )
        
        # 특성 중요도 (트리 모델 또는 선형 모델 기준)
        if 'tree' in models:
            feature_importance = models['tree'].get_feature_importance(feature_cols)
            model_label = f"Tree Model ({tree_type})"
        elif 'linear' in models:
            feature_importance = models['linear'].get_feature_importance(feature_cols)
            model_label = f"Linear Model ({linear_type})"
        else:
            feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': [1/len(feature_cols)] * len(feature_cols)})
            model_label = f"Neural Model ({neural_type})"
        
        # 특성 중요도 시각화
        importance_path = os.path.join(DASHBOARD_DIR, f'feature_importance_{symbol}.png')
        visualizer.plot_feature_importance(feature_importance, model_name=model_label, save_path=importance_path)
        
        # 평가 결과 시각화
        evaluation_path = os.path.join(DASHBOARD_DIR, f'model_evaluation_{symbol}.html')
        evaluator.generate_report(test_data[target_col], predictions, evaluation_path)
        
        return jsonify({
            'success': True, 
            'evaluation': evaluation_results,
            'feature_importance_path': importance_path,
            'evaluation_path': evaluation_path
        })
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict_stock():
    """주가 예측 API"""
    try:
        # 요청 데이터 가져오기
        data = json.loads(request.data.decode('utf-8'))
        symbol = data.get('symbol')
        model_type = data.get('model_type', 'linear')
        
        # 모델 타입 유효성 검사
        if model_type not in ['linear', 'tree', 'neural']:
            return jsonify({'success': False, 'error': 'Invalid model type. Choose from linear, tree, or neural.'})
        
        # 데이터가 있는지 확인
        if symbol not in stock_data:
            return jsonify({'success': False, 'error': f'No data available for {symbol}. Please fetch data first.'})
        
        # 모델이 학습되었는지 확인
        if not models or model_type not in models:
            # 모델 학습 필요
            train_status = train_models()
            if not train_status['success']:
                return jsonify(train_status)
        
        # 예측 수행
        df = stock_data[symbol]
        feature_cols = [col for col in df.columns if col not in ['Returns']]
        model = models[model_type]
        
        # 모델 유형에 따라 예측 수행
        predictions = model.predict(df[feature_cols])
        
        # 결과 시각화
        model_info = model_type
        if model_type == 'linear':
            model_info = f"{model_type} ({data.get('linear_type', 'linear')})"
        elif model_type == 'tree':
            model_info = f"{model_type} ({data.get('tree_type', 'random_forest')})"
        elif model_type == 'neural':
            model_info = f"{model_type} ({data.get('neural_type', 'lstm')})"
        
        # 예측 결과 시각화 
        global evaluator
        if evaluator is None:
            evaluator = ModelEvaluator()
        
        # 예측값이 스칼라인 경우 배열로 변환
        if np.isscalar(predictions):
            predictions = np.array([predictions])
        
        # 모델별 예측값 딕셔너리 생성
        predictions_dict = {model_info: predictions}
        
        # 평가 및 시각화
        target_col = 'Returns'
        
        # 예측 결과 시각화
        prediction_path = os.path.join(DASHBOARD_DIR, f'prediction_{symbol}_{model_info}.png')
        df_target = df[target_col].values
        
        # 길이가 다른 경우 조정
        min_len = min(len(predictions), len(df_target))
        predictions_adjusted = predictions[:min_len]
        df_target_adjusted = df_target[:min_len]
        
        # 예측 시각화 저장
        evaluator.plot_predictions(
            df.index[-min_len:], 
            df_target_adjusted, 
            predictions_adjusted, 
            model_name=model_info, 
            save_path=prediction_path
        )
        
        # 평가 지표 계산
        metrics = evaluator.calculate_metrics(df_target_adjusted, predictions_adjusted)
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'prediction_path': prediction_path
        })
        
    except Exception as e:
        logger.error(f"Error predicting stock price: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dashboard/<path:filename>')
def serve_dashboard(filename):
    """대시보드 파일 제공"""
    return send_from_directory(DASHBOARD_DIR, filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """정적 파일 제공"""
    return send_from_directory(app.static_folder, filename)

def create_app():
    """앱 생성 및 설정"""
    # 템플릿 디렉토리 생성
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # 정적 디렉토리 생성
    static_dir = Path('static')
    static_dir.mkdir(exist_ok=True)
    
    # 기본 템플릿 생성
    with open(templates_dir / 'index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container-fluid">
        <header class="py-3 mb-4 border-bottom">
            <div class="container d-flex flex-wrap justify-content-center">
                <h1 class="text-center">Stock Analysis Dashboard</h1>
            </div>
        </header>

        <div class="row">
            <div class="col-md-3">
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="card-title">Data Controls</h5>
                    </div>
                    <div class="card-body">
                        <form id="dataForm">
                            <div class="mb-3">
                                <label for="symbol" class="form-label">Stock Symbol</label>
                                <input type="text" class="form-control" id="symbol" value="AAPL">
                            </div>
                            <div class="mb-3">
                                <label for="startDate" class="form-label">Start Date</label>
                                <input type="date" class="form-control" id="startDate">
                            </div>
                            <div class="mb-3">
                                <label for="endDate" class="form-label">End Date</label>
                                <input type="date" class="form-control" id="endDate">
                            </div>
                            <button type="button" class="btn btn-primary" id="fetchData">Fetch Data</button>
                        </form>
                    </div>
                </div>

                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="card-title">Model Controls</h5>
                    </div>
                    <div class="card-body">
                        <button type="button" class="btn btn-success mb-3" id="trainModels">Train Models</button>
                        <div class="mb-3">
                            <label for="modelType" class="form-label">Model Type</label>
                            <select class="form-select" id="modelType">
                                <option value="linear">Linear Model</option>
                                <option value="tree" selected>Tree Model</option>
                                <option value="neural">Neural Network</option>
                            </select>
                        </div>
                        <div class="mb-3" id="linearModelOptions" style="display:none;">
                            <label for="linearType" class="form-label">Linear Model Type</label>
                            <select class="form-select" id="linearType">
                                <option value="linear">Basic Linear Regression</option>
                                <option value="ridge">Ridge Regression</option>
                                <option value="lasso">Lasso Regression</option>
                                <option value="elastic_net">ElasticNet Regression</option>
                            </select>
                        </div>
                        <button type="button" class="btn btn-info" id="predict">Predict</button>
                    </div>
                </div>
            </div>

            <div class="col-md-9">
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="card-title">Dashboard</h5>
                    </div>
                    <div class="card-body">
                        <div id="dashboardFrame" style="width: 100%; height: 800px; border: none;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/static/script.js"></script>
</body>
</html>
        ''')
    
    # CSS 파일 생성
    with open(static_dir / 'style.css', 'w') as f:
        f.write('''
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
}

.card {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    margin-bottom: 20px;
}

.card-header {
    background-color: #f1f3f5;
    border-bottom: 1px solid #dee2e6;
    border-radius: 8px 8px 0 0;
}

.card-title {
    margin-bottom: 0;
    color: #495057;
}

.btn-primary {
    background-color: #4361ee;
    border-color: #4361ee;
}

.btn-success {
    background-color: #2ec4b6;
    border-color: #2ec4b6;
}

.btn-info {
    background-color: #4cc9f0;
    border-color: #4cc9f0;
    color: white;
}

.form-control:focus, .form-select:focus {
    border-color: #4361ee;
    box-shadow: 0 0 0 0.25rem rgba(67, 97, 238, 0.25);
}
        ''')
    
    # JavaScript 파일 생성
    with open(static_dir / 'script.js', 'w') as f:
        f.write('''
document.addEventListener('DOMContentLoaded', function() {
    // 날짜 기본값 설정
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    
    document.getElementById('startDate').value = oneYearAgo.toISOString().split('T')[0];
    document.getElementById('endDate').value = today.toISOString().split('T')[0];
    
    // 데이터 가져오기
    document.getElementById('fetchData').addEventListener('click', function() {
        const symbol = document.getElementById('symbol').value;
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;
        
        fetch('/api/fetch_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                start_date: startDate,
                end_date: endDate
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 대시보드 표시
                const dashboardPath = data.data.dashboard_path;
                document.getElementById('dashboardFrame').src = dashboardPath;
                
                alert('Data fetched successfully!');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while fetching data.');
        });
    });
    
    // 모델 학습
    document.getElementById('trainModels').addEventListener('click', function() {
        const symbol = document.getElementById('symbol').value;
        
        fetch('/api/train_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 평가 결과 표시
                const evaluationPath = data.evaluation_path;
                document.getElementById('dashboardFrame').src = evaluationPath;
                
                alert('Models trained successfully!');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while training models.');
        });
    });
    
    // 예측
    document.getElementById('predict').addEventListener('click', function() {
        const symbol = document.getElementById('symbol').value;
        const modelType = document.getElementById('modelType').value;
        
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                model_type: modelType
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 예측 결과 표시
                const visualizationPath = data.prediction_path;
                document.getElementById('dashboardFrame').src = visualizationPath;
                
                alert('Prediction completed successfully!');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while making predictions.');
        });
    });
});
        ''')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000) 