# 고급 자산 가격 변동 분석 프로젝트

이 프로젝트는 다양한 머신러닝 모델을 사용하여 주식 가격 변동을 분석하고 예측하는 고급 파이썬 기반 애플리케이션입니다.

## 주요 기능

- 다중 주식 데이터 수집 및 전처리
- 다양한 기술적 지표 계산
- 여러 머신러닝 모델 비교 (선형 회귀, 랜덤 포레스트, XGBoost, LSTM 등)
- 모델 성능 평가 및 하이퍼파라미터 튜닝
- 대시보드를 통한 결과 시각화
- 포트폴리오 최적화 및 리스크 분석

## 프로젝트 구조

```
advanced_stock_analysis/
├── data/                  # 데이터 저장 및 처리
│   ├── data_loader.py     # 데이터 로딩 및 전처리
│   └── feature_engineering.py  # 특성 엔지니어링
├── models/                # 모델 구현
│   ├── linear_models.py   # 선형 회귀 모델
│   ├── tree_models.py     # 트리 기반 모델
│   ├── neural_models.py   # 신경망 모델
│   └── model_evaluation.py  # 모델 평가
├── utils/                 # 유틸리티 함수
│   ├── visualization.py   # 시각화 도구
│   └── metrics.py         # 평가 지표
├── web/                   # 웹 인터페이스
│   ├── app.py             # Flask/Dash 애플리케이션
│   └── dashboard.py       # 대시보드 구현
├── main.py                # 메인 실행 파일
└── requirements.txt       # 필요한 패키지 목록
```

## 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 데이터 수집 및 전처리:
```bash
python -m data.data_loader
```

2. 모델 학습 및 평가:
```bash
python main.py --mode train
```

3. 웹 대시보드 실행:
```bash
python -m web.app
```

## 모델 설명

1. **선형 회귀 모델**
   - 단순 선형 회귀
   - 다중 선형 회귀
   - 정규화 선형 회귀 (Ridge, Lasso, Elastic Net)

2. **트리 기반 모델**
   - 랜덤 포레스트
   - XGBoost
   - LightGBM

3. **신경망 모델**
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)
   - CNN-LSTM 하이브리드

## 평가 지표

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (R-squared)
- MAPE (Mean Absolute Percentage Error)
- Sharpe Ratio (포트폴리오 성과 평가) 