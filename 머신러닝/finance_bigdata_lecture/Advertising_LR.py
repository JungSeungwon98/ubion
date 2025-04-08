import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# scikit-learn에서 데이터 스케일링을 위한 라이브러리 임포트
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. CSV 파일을 읽어 DataFrame 생성
# index_col=0: CSV에 인덱스(예: "Unnamed: 0")가 포함되어 있다면 제거합니다.
# encoding="utf-8": 파일 인코딩에 맞게 조절 (필요 시 "cp949" 등으로 변경)
df = pd.read_csv("data/outlier_removed.csv", index_col=0, encoding="utf-8")

# 데이터의 일부를 확인합니다.
print("전체 데이터 샘플 (첫 5행):")
print(df.head())

# 예시: 'Sales' 컬럼을 종속 변수(y)로, 나머지 컬럼들을 독립 변수(X)로 설정
X = df.drop('sales', axis=1)  # axis=1은 열을 삭제하는 것을 의미합니다.
y = df['sales']

# 2. 학습셋과 평가셋 (테스트셋)으로 분할
# test_size=0.2: 전체 데이터 중 20%를 평가셋으로 사용합니다.
# random_state=42: 결과 재현성을 위해 난수 seed를 설정합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 분할된 데이터셋의 크기를 출력하여 확인합니다.
print("\n학습셋 크기:", X_train.shape)
print("평가셋 크기:", X_test.shape)

scaler_minmax = MinMaxScaler()
scaler_minmax.fit(X_train)
scaled_X_train = scaler_minmax.transform(X_train)
scaled_X_test = scaler_minmax.transform(X_test)

# 4. 회귀 모델 정의
# 모델별 하이퍼파라미터(예: Lasso, ElasticNet의 alpha, l1_ratio)는 상황에 맞게 조정할 수 있습니다.
lr_model = LinearRegression()
lasso_model = Lasso(alpha=0.1, random_state=42)
ridge_model = Ridge(alpha=1.0, random_state=42)
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# 모델들을 딕셔너리에 저장
models = {
    "Linear Regression": lr_model,
    "Lasso": lasso_model,
    "Ridge": ridge_model,
    "ElasticNet": elastic_model
}

# 5. 각 모델 학습 및 예측, 평가
results = {}

for name, model in models.items():
    # 학습
    model.fit(scaled_X_train, y_train)
    
    # 예측
    y_pred = model.predict(scaled_X_test)
    
    # 평가: 평균제곱오차(MSE)와 결정계수(R²)를 계산합니다.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 결과 저장 및 출력
    results[name] = {"MSE": mse, "R2": r2}
    print(f"{name}: MSE = {mse:.2f}, R2 = {r2:.2f}")