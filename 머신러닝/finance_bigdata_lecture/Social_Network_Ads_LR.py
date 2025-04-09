import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 불러오기: 첨부된 minmax scaling 파일
data_path = 'data/Social_Network_Ads_outlier_removed.csv'
df = pd.read_csv(data_path)

print("데이터 크기:", df.shape)
print("컬럼 목록:", df.columns.tolist())

# LabelEncoder를 이용한 숫자형 카테고리 변환
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# 2. 피처와 타깃 변수 분리
# 여기서는 'Purchased' 컬럼이 예측할 타깃 변수라고 가정합니다.
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# 3. 학습과 테스트 데이터 셋 분리 (예: 70% 학습, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_1 = X_train.drop(['User ID', 'Gender'], axis=1)
X_test_1 = X_test.drop(['User ID', 'Gender'], axis=1)
scaler_minmax = MinMaxScaler()
X_train_minmax_scaled = scaler_minmax.fit_transform(X_train_1)
X_test_minmax_scaled = scaler_minmax.transform(X_test_1)

# 4. 로지스틱 회귀 모델 초기화
# API에 명시된 파라미터를 그대로 사용합니다.
model = LogisticRegression(
    penalty="l2",      
    dual=False,
    tol=0.0001,
    C=10,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=42,
    solver='lbfgs',
    max_iter=1000,
    multi_class='deprecated',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None
)

# 5. 모델 학습
model.fit(X_train_minmax_scaled, y_train)

# 6. 테스트 데이터에 대한 예측 수행
y_pred = model.predict(X_test_minmax_scaled)

# 7. 예측 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("정확도(Accuracy): {:.2f}".format(accuracy))
print("roc_auc_score: {:.2f}".format(roc_auc))
print("혼동 행렬(Confusion Matrix):")
print(conf_matrix)
print("분류 보고서(Classification Report):")
print(class_report)