import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import yfinance as yf
import logging
import os
from pathlib import Path
from functools import lru_cache
import json
import traceback
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# 대시보드 데이터 저장 경로
DASHBOARD_DIR = Path('dashboard')
DASHBOARD_DIR.mkdir(exist_ok=True)

class StockDashboard:
    def __init__(self):
        """대시보드 클래스 초기화"""
        import os
        
        self.app = Dash(__name__, 
                      external_stylesheets=[dbc.themes.BOOTSTRAP], 
                      suppress_callback_exceptions=True,
                      assets_folder=os.path.join(os.getcwd(), 'dashboard'))
        
        # 대시보드 폴더 생성
        dashboard_dir = os.path.join(os.getcwd(), 'dashboard')
        if not os.path.exists(dashboard_dir):
            os.makedirs(dashboard_dir)
            print(f"Created dashboard directory at {dashboard_dir}")
        
        # 정적 파일 경로 명시적으로 설정
        self.app.server.static_folder = dashboard_dir
        self.app.server.static_url_path = '/static'
        
        # 추가 정적 파일 경로 설정
        @self.app.server.route('/dashboard/<path:filename>')
        def serve_dashboard_file(filename):
            from flask import send_from_directory
            return send_from_directory(dashboard_dir, filename)
        
        # index.html 파일 생성 (필요시)
        index_html = os.path.join(dashboard_dir, 'dashboard.html')
        if not os.path.exists(index_html):
            with open(index_html, 'w') as f:
                f.write('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Stock Analysis Dashboard</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <link rel="stylesheet" href="/static/style.css">
                </head>
                <body>
                    <div id="root">
                        <h1>Stock Analysis Dashboard</h1>
                        <p>This is a static page. Please use the main dashboard at <a href="/">Dashboard Home</a>.</p>
                    </div>
                </body>
                </html>
                ''')
            
            # CSS 파일도 생성
            css_file = os.path.join(dashboard_dir, 'style.css')
            if not os.path.exists(css_file):
                with open(css_file, 'w') as f:
                    f.write('''
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }
                    h1 {
                        color: #333;
                    }
                    a {
                        color: #0066cc;
                        text-decoration: none;
                    }
                    a:hover {
                        text-decoration: underline;
                    }
                    ''')
        
        self.data = {}
        self.models = {}
        self.evaluator = None
        self.visualizer = None
        
        # 레이아웃 설정
        self.app.layout = self.create_layout()
        
        # 콜백 등록
        self.register_callbacks()
    
    def create_layout(self):
        """대시보드 레이아웃 생성"""
        return dbc.Container([
            # 데이터를 저장할 dcc.Store 추가
            dcc.Store(id="data-store", storage_type="memory"),
            
            dbc.Row([
                dbc.Col(html.H1("Stock Analysis Dashboard", className="text-center my-4"), width=12)
            ]),
            
            dbc.Row([
                # 왼쪽 컨트롤 패널
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Data Controls"),
                        dbc.CardBody([
                            dbc.Form([
                                dbc.Row([
                                    dbc.Label("Stock Symbol"),
                                    dbc.Input(id="symbol-input", type="text", value="AAPL", className="mb-3")
                                ]),
                                dbc.Row([
                                    dbc.Label("Start Date"),
                                    dbc.Input(id="start-date-input", type="date", className="mb-3")
                                ]),
                                dbc.Row([
                                    dbc.Label("End Date"),
                                    dbc.Input(id="end-date-input", type="date", className="mb-3")
                                ]),
                                dbc.Button("Fetch Data", id="fetch-data-button", color="primary", className="mt-3")
                            ])
                        ])
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Model Controls"),
                        dbc.CardBody([
                            dbc.Button("Train Models", id="train-models-button", color="success", className="mb-3"),
                            dbc.Row([
                                dbc.Label("Model Type"),
                                dbc.Select(
                                    id="model-type-select",
                                    options=[
                                        {"label": "Linear Model", "value": "linear"},
                                        {"label": "Tree Model", "value": "tree"},
                                        {"label": "Neural Network", "value": "neural"}
                                    ],
                                    value="tree",
                                    className="mb-3"
                                )
                            ]),
                            dbc.Button("Predict", id="predict-button", color="info", className="mt-3")
                        ])
                    ])
                ], width=3),
                
                # 오른쪽 대시보드 패널
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="Price History", tab_id="price-tab"),
                        dbc.Tab(label="Technical Indicators", tab_id="indicators-tab"),
                        dbc.Tab(label="Model Evaluation", tab_id="evaluation-tab"),
                        dbc.Tab(label="Predictions", tab_id="predictions-tab")
                    ], id="tabs", active_tab="price-tab"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id="tab-content")
                        ])
                    ], className="mt-3")
                ], width=9)
            ])
        ], fluid=True)
    
    def register_callbacks(self):
        """콜백 함수 등록"""
        
        # 날짜 기본값 설정
        @self.app.callback(
            [Output("start-date-input", "value"),
             Output("end-date-input", "value")],
            Input("tabs", "active_tab")
        )
        def set_default_dates(active_tab):
            today = datetime.now()
            one_year_ago = today - timedelta(days=365)
            return one_year_ago.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        
        # 데이터 가져오기 및 탭 콘텐츠 업데이트를 하나의 콜백으로 통합
        @self.app.callback(
            [Output("tab-content", "children"),
             Output("data-store", "data")],
            [Input("fetch-data-button", "n_clicks"),
             Input("tabs", "active_tab")],
            [State("symbol-input", "value"),
             State("start-date-input", "value"),
             State("end-date-input", "value"),
             State("data-store", "data")]
        )
        def update_content(n_clicks, active_tab, symbol, start_date, end_date, data_store):
            from dash import callback_context
            
            # 콜백 컨텍스트에서 어떤 Input이 트리거되었는지 확인
            ctx = callback_context
            trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
            
            # 초기 상태
            if not ctx.triggered:
                return html.Div("Please fetch data to begin analysis."), {}
            
            # 데이터 가져오기 버튼이 클릭된 경우
            if trigger_id == "fetch-data-button.n_clicks" and n_clicks:
                try:
                    # yfinance에서 직접 데이터 가져오기
                    import yfinance as yf
                    import numpy as np
                    import re
                    
                    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
                    
                    # 데이터 다운로드 - auto_adjust=True로 설정하여 수정주가 사용
                    stock_data = yf.download(
                        symbol, 
                        start=start_date, 
                        end=end_date,
                        auto_adjust=True,  # 수정주가 사용
                        progress=False
                    )
                    
                    if stock_data.empty:
                        logger.error(f"No data available for {symbol}")
                        return html.Div(f"데이터를 찾을 수 없습니다: {symbol}. 심볼과 날짜 범위를 확인해주세요."), {}
                    
                    logger.info(f"Downloaded data shape: {stock_data.shape}")
                    logger.info(f"Downloaded data columns: {stock_data.columns.tolist()}")
                    logger.info(f"Downloaded data sample: \n{stock_data.head(3)}")
                    
                    # 데이터 타입 확인 및 변환
                    for col in stock_data.columns:
                        logger.info(f"Column {col} data type: {stock_data[col].dtype}")
                        
                        # 문자열이 포함된 데이터 확인 및 숫자로 변환
                        if stock_data[col].dtype == 'object':
                            logger.info(f"Converting {col} from object to float")
                            
                            # 예: "Ticker AAPL 168.24 Name: 2024-04-08 00:00:00, dtype: float64" 형식에서 숫자만 추출
                            def extract_number(val):
                                if isinstance(val, str):
                                    # 정규식으로 숫자 추출 (소수점 포함)
                                    match = re.search(r'(\d+\.\d+)', val)
                                    if match:
                                        return float(match.group(1))
                                return val
                            
                            # 각 값을 숫자로 변환 시도
                            stock_data[col] = stock_data[col].apply(extract_number)
                    
                    # 데이터프레임 생성
                    df = stock_data.copy()
                    
                    # 필수 컬럼 확인
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    # 데이터 숫자 타입으로 변환 확인
                    for col in df.columns:
                        try:
                            if df[col].dtype == 'object':
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                logger.info(f"Converted {col} to numeric")
                        except Exception as e:
                            logger.error(f"Error converting {col} to numeric: {str(e)}")
                    
                    # 모든 데이터가 숫자인지 확인
                    for col in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            logger.warning(f"Column {col} is not numeric: {df[col].dtype}")
                            logger.info(f"Sample values: {df[col].head(3)}")
                    
                    # 결측치 및 무한값 처리
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    
                    # 데이터 확인 로깅
                    logger.info(f"Processed data shape: {df.shape}")
                    logger.info(f"Processed columns: {df.columns.tolist()}")
                    logger.info(f"Data types after processing: {df.dtypes}")
                    
                    # 특성 엔지니어링
                    try:
                        logger.info("Starting feature engineering")
                        logger.info(f"DataFrame columns before feature engineering: {df.columns.tolist()}")
                        
                        engineer = FeatureEngineer()
                        df = engineer.process(df)  # process 메서드 사용
                        
                        logger.info("Feature engineering completed successfully")
                        logger.info(f"DataFrame shape after feature engineering: {df.shape}")
                    except Exception as e:
                        logger.error(f"Error during feature engineering: {str(e)}")
                        # 특성 엔지니어링 실패해도 기본 데이터로 진행
                    
                    # 전역 변수에 저장
                    self.data[symbol] = df
                    
                    # 시각화 도구 초기화
                    self.visualizer = StockVisualizer()
                    
                    try:
                        # 데이터 샘플 자세히 확인
                        logger.info(f"Chart data head:\n{df.head(3)}")
                        logger.info(f"Chart data columns type: {type(df.columns)}")
                        
                        # DataFrame 컬럼이 MultiIndex인 경우 처리
                        if isinstance(df.columns, pd.MultiIndex):
                            logger.info("Converting MultiIndex columns to string")
                            # 컬럼을 문자열로 변환
                            string_columns = []
                            for col in df.columns:
                                if isinstance(col, tuple):
                                    string_columns.append("_".join([str(c) for c in col if c]))
                                else:
                                    string_columns.append(str(col))
                            df.columns = string_columns
                            logger.info(f"Converted columns: {df.columns.tolist()}")
                        
                        # 데이터가 비어있거나 인덱스가 없는지 확인
                        if df.empty:
                            logger.warning("DataFrame is empty")
                            return html.Div("데이터가 비어있습니다. 다른 심볼이나 날짜 범위를 선택해주세요."), {}
                        
                        if len(df) == 0:
                            logger.warning("DataFrame has no rows")
                            return html.Div("데이터가 비어있습니다. 다른 심볼이나 날짜 범위를 선택해주세요."), {}
                        
                        # 가장 기본적인 HTML 출력만 생성
                        # 간단한 테이블로 데이터 표시
                        try:
                            # 데이터 기본 정보 표시
                            data_info = [
                                html.H4(f"{symbol} 주식 데이터"),
                                html.P(f"시작일: {start_date}, 종료일: {end_date}"),
                                html.P(f"데이터 크기: {df.shape[0]} 행 × {df.shape[1]} 열"),
                                html.P(f"수집된 데이터: {', '.join(df.columns.tolist())}")
                            ]
                            
                            # 데이터가 충분히 있을 때만 표시
                            if len(df) >= 5:
                                # 데이터 테이블 생성
                                data_table = html.Table([
                                    html.Thead(
                                        html.Tr([html.Th("날짜")] + [html.Th(col) for col in df.columns[:5]])
                                    ),
                                    html.Tbody([
                                        html.Tr([html.Td(df.index[i].strftime('%Y-%m-%d') if hasattr(df.index[i], 'strftime') else str(df.index[i]))] + 
                                                [html.Td(str(round(df.iloc[i][col], 2)) if pd.notna(df.iloc[i][col]) else "N/A") 
                                                 for col in df.columns[:5]])
                                        for i in range(min(5, len(df)))
                                    ])
                                ], className="table table-striped")
                                
                                data_html = html.Div(data_info + [html.Hr(), data_table])
                            else:
                                data_html = html.Div(data_info + [
                                    html.Hr(),
                                    html.P("데이터 행이 부족하여 테이블을 표시할 수 없습니다.")
                                ])
                        except Exception as e:
                            logger.error(f"Error creating data table: {str(e)}")
                            # 오류 발생 시 기본 정보만 표시
                            data_html = html.Div([
                                html.H4(f"{symbol} 주식 데이터"),
                                html.P(f"데이터 크기: {df.shape[0]} 행 × {df.shape[1]} 열"),
                                html.Hr(),
                                html.P(f"테이블 생성 중 오류 발생: {str(e)}")
                            ])
                        
                        # 데이터 저장
                        data_dict = {
                            'symbol': symbol,
                            'start_date': start_date,
                            'end_date': end_date,
                            'rows': len(df),
                            'columns': list(df.columns)
                        }
                        
                        # 차트는 생성하지 않고 데이터만 표시
                        return data_html, data_dict
                        
                    except Exception as e:
                        import traceback
                        logger.error(f"Error creating chart: {str(e)}")
                        logger.error(traceback.format_exc())
                        return html.Div([
                            html.H4("차트 생성 중 오류 발생"),
                            html.Pre(traceback.format_exc())
                        ]), {}
                
                except Exception as e:
                    logger.error(f"Error fetching data: {str(e)}")
                    return html.Div(f"오류: {str(e)}"), {}
            
            # 탭 변경이 트리거된 경우
            elif trigger_id == "tabs.active_tab":
                # 데이터가 없는 경우
                if not data_store or symbol not in self.data:
                    return html.Div("Please fetch data to begin analysis."), data_store or {}
                
                df = self.data[symbol]
                
                # DataFrame 컬럼이 MultiIndex인 경우 처리
                if isinstance(df.columns, pd.MultiIndex):
                    logger.info("Converting MultiIndex columns in tab change")
                    # 컬럼을 문자열로 변환
                    string_columns = []
                    for col in df.columns:
                        if isinstance(col, tuple):
                            string_columns.append("_".join([str(c) for c in col if c]))
                        else:
                            string_columns.append(str(col))
                    df.columns = string_columns
                
                # 모든 탭에 대해 간단한 데이터 정보만 표시
                if active_tab == "price-tab":
                    return html.Div([
                        html.H4(f"{symbol} 가격 데이터"),
                        html.P(f"데이터 기간: {min(df.index).strftime('%Y-%m-%d')} ~ {max(df.index).strftime('%Y-%m-%d')}"),
                        html.P(f"데이터 행 수: {len(df)}"),
                        html.Hr(),
                        html.Pre(df[df.columns[:5]].head(15).to_string())
                    ]), data_store
                
                elif active_tab == "indicators-tab":
                    # 기술적 지표 관련 컬럼 필터링
                    indicator_cols = [col for col in df.columns if any(ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD', 'BB_'])]
                    
                    return html.Div([
                        html.H4(f"{symbol} 기술적 지표"),
                        html.P(f"사용 가능한 지표: {', '.join(indicator_cols)}"),
                        html.Hr(),
                        html.Pre(df[indicator_cols[:5]].head(10).to_string() if indicator_cols else "기술적 지표 데이터가 없습니다.")
                    ]), data_store
                    
                elif active_tab == "evaluation-tab":
                    return html.Div([
                        html.H4("모델 평가"),
                        html.P("'Train Models' 버튼을 클릭하여 모델을 학습하고 평가하세요."),
                        html.Div(id="evaluation-results", children="모델이 학습되지 않았습니다.")
                    ]), data_store
                
                elif active_tab == "predictions-tab":
                    return html.Div([
                        html.H4("예측 결과"),
                        html.P("'Predict' 버튼을 클릭하여 예측 결과를 확인하세요."),
                        html.Div(id="prediction-results", children="예측이 수행되지 않았습니다.")
                    ]), data_store
                
                else:
                    return html.Div("Unknown tab selected"), data_store
            
            # 기본 반환
            return html.Div("Please fetch data to begin analysis."), data_store or {}
        
        # 모델 학습
        @self.app.callback(
            Output("evaluation-results", "children"),
            [Input("train-models-button", "n_clicks")],
            [State("symbol-input", "value")]
        )
        def train_models(n_clicks, symbol):
            if n_clicks is None or symbol not in self.data:
                return html.Div("Please fetch data and click 'Train Models' to see evaluation results.")
            
            try:
                df = self.data[symbol]
                
                # 데이터에 NaN 값이 있는지 확인
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in the data. Applying imputation.")
                    # NaN 값 처리
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    
                    # 여전히 NaN 값이 있는지 확인
                    remaining_nans = df.isna().sum().sum()
                    if remaining_nans > 0:
                        logger.warning(f"After ffill/bfill, still have {remaining_nans} NaN values. Using mean imputation.")
                        # 각 열의 평균으로 채우기
                        df = df.fillna(df.mean())
                        # 여전히 NaN 값이 있다면 0으로 채우기
                        df = df.fillna(0)
                
                # 데이터 분할
                train_size = int(len(df) * 0.8)
                train_data = df.iloc[:train_size]
                test_data = df.iloc[train_size:]
                
                # 특성 및 타겟 설정
                feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
                
                # Returns 열이 없는 경우 생성
                if 'Returns' not in df.columns and 'Close' in df.columns:
                    logger.info("Creating Returns column")
                    df['Returns'] = df['Close'].pct_change()
                    train_data = df.iloc[:train_size]
                    test_data = df.iloc[train_size:]
                
                target_col = 'Returns'
                
                # 학습 데이터에 타겟 컬럼이 있는지 확인
                if target_col not in train_data.columns:
                    logger.error(f"Target column '{target_col}' not found in data")
                    return html.Div(f"Error: Target column '{target_col}' not found in data")
                
                # 특성 컬럼이 비어 있는지 확인
                if not feature_cols:
                    logger.error("No feature columns available for training")
                    return html.Div("Error: No feature columns available for training")
                
                # 훈련 및 테스트 데이터에 NaN 값이 있는지 다시 확인
                X_train_nans = train_data[feature_cols].isna().sum().sum()
                y_train_nans = train_data[target_col].isna().sum()
                
                if X_train_nans > 0 or y_train_nans > 0:
                    logger.warning(f"Training data still has NaN values: X={X_train_nans}, y={y_train_nans}")
                    logger.info("Applying SimpleImputer to handle NaN values")
                    
                    from sklearn.impute import SimpleImputer
                    
                    # 특성 데이터 처리
                    X_imputer = SimpleImputer(strategy='mean')
                    train_data[feature_cols] = pd.DataFrame(
                        X_imputer.fit_transform(train_data[feature_cols]),
                        columns=feature_cols,
                        index=train_data.index
                    )
                    
                    test_data[feature_cols] = pd.DataFrame(
                        X_imputer.transform(test_data[feature_cols]),
                        columns=feature_cols,
                        index=test_data.index
                    )
                    
                    # 타겟 데이터 처리
                    if y_train_nans > 0:
                        y_imputer = SimpleImputer(strategy='mean')
                        train_data[target_col] = pd.Series(
                            y_imputer.fit_transform(train_data[target_col].values.reshape(-1, 1)).flatten(),
                            index=train_data.index,
                            name=target_col
                        )
                        
                        test_data[target_col] = pd.Series(
                            y_imputer.transform(test_data[target_col].values.reshape(-1, 1)).flatten(),
                            index=test_data.index,
                            name=target_col
                        )
                
                # 모델 학습
                self.models = {}
                
                # 선형 모델
                try:
                    logger.info("Training linear model")
                    linear_model = LinearModel()
                    linear_model.train(train_data[feature_cols], train_data[target_col])
                    self.models['linear'] = linear_model
                    logger.info("Linear model training complete")
                except Exception as e:
                    logger.error(f"Error training linear model: {str(e)}")
                
                # 트리 모델
                try:
                    logger.info("Training tree model")
                    tree_model = TreeModel()
                    tree_model.train(train_data[feature_cols], train_data[target_col])
                    self.models['tree'] = tree_model
                    logger.info("Tree model training complete")
                except Exception as e:
                    logger.error(f"Error training tree model: {str(e)}")
                
                # 신경망 모델
                try:
                    logger.info("Training neural network model")
                    neural_model = NeuralModel()
                    neural_model.train(train_data[feature_cols], train_data[target_col])
                    self.models['neural'] = neural_model
                    logger.info("Neural network model training complete")
                except Exception as e:
                    logger.error(f"Error training neural model: {str(e)}")
                
                # 모델이 하나도 없는 경우
                if not self.models:
                    return html.Div("Error: Failed to train any models")
                
                # 모델 평가
                self.evaluator = ModelEvaluator()
                
                # 예측 수행
                predictions = {}
                for name, model in self.models.items():
                    try:
                        predictions[name] = model.predict(test_data[feature_cols])
                        logger.info(f"Generated predictions using {name} model")
                    except Exception as e:
                        logger.error(f"Error predicting with {name} model: {str(e)}")
                
                # 모델이 예측 실패한 경우
                if not predictions:
                    return html.Div("Error: Failed to generate predictions with any model")
                
                # 평가 결과
                evaluation_results = self.evaluator.evaluate_models(
                    test_data[target_col], 
                    predictions
                )
                
                # 특성 중요도 (가능한 경우)
                feature_importance = None
                if 'tree' in self.models:
                    feature_importance = self.models['tree'].get_feature_importance(feature_cols)
                elif 'linear' in self.models and hasattr(self.models['linear'], 'get_feature_importance'):
                    feature_importance = self.models['linear'].get_feature_importance(feature_cols)
                
                # 결과 표시
                evaluation_components = [html.H4("Model Performance Metrics")]
                
                # 평가 테이블 생성
                if evaluation_results:
                    # 평가 결과 테이블
                    evaluation_table = dbc.Table([
                        html.Thead(html.Tr([html.Th("Metric")] + [
                            html.Th(name.capitalize()) for name in evaluation_results.keys()
                        ])),
                        html.Tbody([
                            html.Tr([html.Td(metric)] + [
                                html.Td(f"{evaluation_results[model].get(metric, 'N/A'):.6f}" 
                                        if isinstance(evaluation_results[model].get(metric, 'N/A'), (int, float)) 
                                        else "N/A")
                                for model in evaluation_results.keys()
                            ])
                            for metric in ['mse', 'rmse', 'mae', 'r2', 'sharpe_ratio', 'max_drawdown']
                            if any(metric in evaluation_results[model] for model in evaluation_results.keys())
                        ])
                    ], bordered=True, hover=True, responsive=True, striped=True)
                    
                    evaluation_components.append(evaluation_table)
                
                # 특성 중요도 추가 (가능한 경우)
                if feature_importance is not None and len(feature_importance) > 0:
                    try:
                        # 특성 중요도 시각화
                        importance_fig = px.bar(
                            feature_importance.sort_values('importance', ascending=True).tail(10),
                            x='importance',
                            y='feature',
                            title='Top 10 Feature Importance',
                            orientation='h'
                        )
                        
                        evaluation_components.extend([
                            html.H4("Feature Importance"),
                            dcc.Graph(figure=importance_fig)
                        ])
                    except Exception as e:
                        logger.error(f"Error creating feature importance visualization: {str(e)}")
                
                return html.Div(evaluation_components)
            
            except Exception as e:
                logger.error(f"Error training models: {str(e)}")
                import traceback
                trace = traceback.format_exc()
                logger.error(trace)
                return html.Div([
                    html.H4("Error Training Models"),
                    html.P(str(e)),
                    html.Pre(trace)
                ])
        
        # 예측
        @self.app.callback(
            Output("prediction-results", "children"),
            [Input("predict-button", "n_clicks")],
            [State("symbol-input", "value"),
             State("model-type-select", "value")]
        )
        def predict(n_clicks, symbol, model_type):
            if n_clicks is None or symbol not in self.data or not self.models or model_type not in self.models:
                return html.Div("Please fetch data, train models, and select a model type to see predictions.")
            
            try:
                df = self.data[symbol]
                model = self.models[model_type]
                
                # 특성 설정
                feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
                
                logger.info(f"Feature columns for prediction: {len(feature_cols)} columns")
                logger.info(f"Feature column names: {feature_cols}")
                
                if len(feature_cols) == 0:
                    return html.Div("Error: No feature columns available for prediction")
                
                # NaN 값 확인 및 처리
                nan_count = df[feature_cols].isna().sum().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in prediction data. Applying imputation.")
                    
                    # df_clean 초기화 - 원본 데이터 복사
                    df_clean = df[feature_cols].copy()
                    
                    # 각 열마다 NaN 값을 처리
                    for col in df_clean.columns:
                        if df_clean[col].isna().any():
                            # forward fill, backward fill 적용
                            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                            # 여전히 NaN이 남아있으면 평균값으로 대체
                            if df_clean[col].isna().any():
                                mean_val = df_clean[col].mean()
                                if pd.isna(mean_val):  # 전체 열이 NaN인 경우
                                    df_clean[col] = df_clean[col].fillna(0)
                                else:
                                    df_clean[col] = df_clean[col].fillna(mean_val)
                else:
                    df_clean = df[feature_cols].copy()
                
                # 모든 열에 여전히 NaN 값이 있는지 확인
                remaining_nans = df_clean.isna().sum().sum()
                if remaining_nans > 0:
                    logger.warning(f"Still found {remaining_nans} NaN values after basic imputation. Filling with zeros.")
                    df_clean = df_clean.fillna(0)
                
                # 예측 수행
                logger.info(f"Predicting with {model_type} model on {len(df_clean)} rows of data")
                try:
                    predictions = model.predict(df_clean)
                    logger.info(f"Prediction complete: generated {len(predictions)} predictions")
                except Exception as e:
                    logger.error(f"Error during model prediction: {str(e)}")
                    # 더 간단한 특성셋으로 재시도
                    logger.info("Retrying with basic features only")
                    # 훈련에 사용된 특성만 선택
                    if hasattr(model, 'get_feature_names') and callable(model.get_feature_names):
                        used_features = model.get_feature_names()
                        logger.info(f"Model was trained on {len(used_features)} features: {used_features}")
                        # 훈련에 사용된 특성만 있는지 확인
                        available_features = [f for f in used_features if f in df.columns]
                        if available_features:
                            logger.info(f"Using {len(available_features)} available features: {available_features}")
                            df_clean = df[available_features].fillna(0)
                            predictions = model.predict(df_clean)
                        else:
                            return html.Div("Error: None of the features used in training are available in the current data")
                    else:
                        # 기본 특성만 사용해보기 
                        basic_features = [col for col in df.columns if col in ['Open', 'High', 'Low', 'Volume']]
                        if basic_features:
                            logger.info(f"Fallback to basic features: {basic_features}")
                            df_clean = df[basic_features].fillna(0)
                            predictions = model.predict(df_clean)
                        else:
                            return html.Div("Error: Could not find suitable features for prediction")
                
                # Returns 열이 없는 경우 생성
                if 'Returns' not in df.columns and 'Close' in df.columns:
                    logger.info("Creating Returns column for visualization")
                    df['Returns'] = df['Close'].pct_change()
                
                # 예측 결과 시각화
                fig = make_subplots(rows=1, cols=1)
                
                # 데이터 부족 시 처리
                if len(df) < 3:
                    return html.Div("Not enough data points for visualization")
                
                try:
                    # 실제 데이터 플롯
                    if 'Returns' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['Returns'],
                                name='Actual Returns',
                                line=dict(color='blue')
                            )
                        )
                    
                    # 예측 데이터 플롯
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=predictions,
                            name='Predicted Returns',
                            line=dict(color='red')
                        )
                    )
                    
                    # 레이아웃 설정
                    fig.update_layout(
                        title=f'{symbol} Returns Prediction ({model_type.capitalize()} Model)',
                        xaxis_title='Date',
                        yaxis_title='Returns',
                        height=500
                    )
                except Exception as e:
                    logger.error(f"Error creating plot: {str(e)}")
                    return html.Div(f"Error creating plot: {str(e)}")
                
                # 예측 결과 테이블 (마지막 10개 행)
                try:
                    # 예측 및 실제 값 통합 테이블 생성
                    table_data = []
                    for i in range(min(10, len(df))):
                        idx = -min(10, len(df)) + i
                        row = [
                            df.index[idx].strftime('%Y-%m-%d') if hasattr(df.index[idx], 'strftime') else str(df.index[idx]),
                        ]
                        
                        # 실제 값이 있으면 추가
                        if 'Returns' in df.columns:
                            row.append(f"{df['Returns'].iloc[idx]:.6f}" if not pd.isna(df['Returns'].iloc[idx]) else "N/A")
                        else:
                            row.append("N/A")
                        
                        # 예측 값 추가
                        row.append(f"{predictions[idx]:.6f}" if not pd.isna(predictions[idx]) else "N/A")
                        
                        table_data.append(html.Tr([html.Td(cell) for cell in row]))
                    
                    prediction_table = dbc.Table([
                        html.Thead(html.Tr([html.Th("Date"), html.Th("Actual Returns"), html.Th("Predicted Returns")])),
                        html.Tbody(table_data)
                    ], bordered=True, hover=True, responsive=True, striped=True)
                except Exception as e:
                    logger.error(f"Error creating table: {str(e)}")
                    prediction_table = html.Div(f"Error creating table: {str(e)}")
                
                return html.Div([
                    dcc.Graph(figure=fig),
                    html.H4("Recent Predictions"),
                    prediction_table
                ])
            
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                import traceback
                trace = traceback.format_exc()
                logger.error(trace)
                return html.Div([
                    html.H4("Error Making Prediction"),
                    html.P(str(e)),
                    html.Pre(trace)
                ])
    
    def run(self, debug=True, host='0.0.0.0', port=8050):
        """대시보드 실행"""
        logger.info(f"Running dashboard on {host}:{port}")
        
        # suppress_callback_exceptions는 Dash 앱 초기화 시에만 설정해야 함 (이미 __init__에서 설정됨)
        # Flask run_simple() 함수에는 전달하지 않음
        self.app.run_server(
            debug=debug,               # 디버그 모드
            host=host,                 # 호스트
            port=port,                 # 포트
            dev_tools_hot_reload=True, # 자동 리로드
            use_reloader=True          # 파일 변경시 리로드
        )

def main():
    """메인 함수"""
    import os
    
    # 대시보드 폴더 생성
    dashboard_dir = os.path.join(os.getcwd(), 'dashboard')
    if not os.path.exists(dashboard_dir):
        os.makedirs(dashboard_dir)
        print(f"Created dashboard directory at {dashboard_dir}")
    
    dashboard = StockDashboard()
    
    # 대시보드 폴더를 정적 폴더로 설정
    dashboard.app.server.static_folder = dashboard_dir
    
    dashboard.run()

if __name__ == "__main__":
    main() 