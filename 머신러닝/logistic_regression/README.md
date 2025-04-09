# 주식 시장 변동성 예측 시스템

이 프로젝트는 로지스틱 회귀를 사용하여 주식 시장의 변동성 급증을 예측하는 시스템입니다.

## 주요 기능

- 주식 데이터 및 VIX 지수 데이터 수집
- 변동성 계산 및 특징 추출
- 로지스틱 회귀 모델을 통한 변동성 급증 예측
- Streamlit 기반 인터랙티브 대시보드를 통한 시각화
- 모델 성능 평가 및 특성 중요도 분석
- 데이터 다운로드 기능

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Streamlit 설치 (requirements.txt에 포함되어 있지만, 별도로 설치하려면):
```bash
pip install streamlit
```

## 사용 방법

### 모델 학습
```bash
python main.py --mode train --symbol AAPL --days 365
```

### Streamlit 대시보드 실행
다음 방법 중 하나를 선택하여 실행할 수 있습니다:

1. `run_app.py` 실행 (가장 간단한 방법):
```bash
python run_app.py
```

2. `main.py`를 통한 실행:
```bash
python main.py --mode streamlit
```

3. 직접 Streamlit 명령어 사용:
```bash
python -m streamlit run app.py
```

## 대시보드 구성

1. **설정 패널**: 주식 심볼, 기간 설정
2. **변동성 차트**: 주가 변동성의 시계열 추이
3. **예측 차트**: 변동성 급증 예측 확률
4. **특성 중요도**: 각 특성이 예측에 미치는 영향
5. **모델 성능 지표**: 정확도 및 분류 보고서
6. **데이터 다운로드**: 분석 결과를 CSV 파일로 다운로드

## 프로젝트 구조

- `data_loader.py`: 데이터 수집 및 전처리
- `preprocessor.py`: 데이터 스케일링 및 분할
- `model.py`: 로지스틱 회귀 모델 구현
- `app.py`: Streamlit 기반 대시보드
- `main.py`: 메인 실행 파일
- `run_app.py`: Streamlit 앱 실행 스크립트

## 의존성

- Python 3.8+
- pandas
- numpy
- scikit-learn
- yfinance
- streamlit
- plotly
- joblib

## 문제 해결

Streamlit 실행 시 문제가 발생하면 다음을 시도해보세요:

1. Streamlit이 제대로 설치되었는지 확인:
```bash
pip show streamlit
```

2. 가상환경을 사용 중이라면 가상환경이 활성화되어 있는지 확인

3. Python 경로가 올바르게 설정되어 있는지 확인

## 라이선스

MIT License 