# Advanced Stock Analysis

고급 주식 분석 프로젝트 - 데이터 수집, 특성 엔지니어링, 모델링, 시각화를 포함한 종합 분석 도구

## 주요 기능

- 주식 데이터 수집 및 전처리
- 기술적 지표 계산 및 특성 엔지니어링
- 다양한 머신러닝 모델 (선형, 트리, 신경망)
- 대화형 대시보드 및 시각화
- CLI, 웹, 대시보드 모드 지원

## 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### CLI 모드

기본 사용법:
```bash
python advanced_stock_analysis/main.py --mode cli --symbol AAPL --start-date 2023-01-01 --end-date 2024-01-01
```

특정 모델 사용:
```bash
python advanced_stock_analysis/main.py --mode cli --symbol AAPL --start-date 2023-01-01 --end-date 2024-01-01 --model linear
```

### 웹 모드

```bash
python advanced_stock_analysis/main.py --mode web --host 0.0.0.0 --port 5000
```

### 대시보드 모드

```bash
python advanced_stock_analysis/main.py --mode dashboard --host 0.0.0.0 --port 8050
```

## 주요 컴포넌트

### 데이터 로더 (`data_loader.py`)
- yfinance API를 사용한 주식 데이터 수집
- 자동 재시도 메커니즘
- 데이터 유효성 검사 및 전처리

### 특성 엔지니어링 (`feature_engineering.py`)
- 기술적 지표 계산 (pandas_ta 라이브러리 사용)
- 시간 기반 특성
- 지연 및 롤링 특성
- 자동 결측값 처리

### 모델링
- 선형 모델 (`linear_models.py`)
- 트리 기반 모델 (`tree_models.py`)
- 신경망 모델 (`neural_models.py`)
- 모델 평가 및 성능 지표

### 시각화 (`visualization.py`)
- 대화형 차트 및 그래프
- 기술적 지표 시각화
- 상관관계 분석
- 종합 대시보드 생성

## 최근 업데이트

- talib에서 pandas_ta로 기술적 지표 계산 라이브러리 변경
- 데이터 로더의 안정성 개선
- 멀티인덱스 컬럼 처리 개선
- NaN 값 처리 로직 강화
- 시각화 컴포넌트 안정성 개선
- 명령행 인수 처리 개선 (하이픈 및 언더스코어 형식 지원)

## 주의사항

- 일부 기능은 인터넷 연결이 필요합니다.
- 실시간 데이터 수집 시 API 제한에 유의하세요.
- 대량의 데이터 처리 시 충분한 메모리가 필요할 수 있습니다.

## 라이선스

MIT License

## 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다. 