import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import platform

# scikit-learn에서 데이터 스케일링을 위한 라이브러리 임포트
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 운영체제에 맞게 한글을 지원하는 폰트를 설정합니다.
if platform.system() == 'Windows':
    # Windows의 경우 'Malgun Gothic'을 많이 사용합니다.
    mpl.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    # macOS의 경우 'AppleGothic'을 사용하거나, 설치된 한글 폰트를 선택합니다.
    mpl.rcParams['font.family'] = 'AppleGothic'
else:
    # Linux의 경우 'NanumGothic' 등 한글 지원 폰트를 사용할 수 있습니다.
    mpl.rcParams['font.family'] = 'NanumGothic'

# 1. 데이터 불러오기
# index_col=0: CSV 저장 시 인덱스가 함께 저장된 경우 제거
# encoding 옵션은 파일 인코딩에 맞게 조정 (예: 'utf-8' 또는 'cp949')
df = pd.read_csv("data/outlier_removed.csv", index_col=0, encoding="utf-8")

# 2. 전처리: 숫자형 변수 선택 (스케일링은 보통 숫자형 변수에 적용)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\n스케일링 대상 숫자형 컬럼:")
print(numeric_cols.tolist())

# 3. 스케일링 전 데이터의 분포 시각화 (Box Plot)
plt.figure(figsize=(14, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("스케일링 전 데이터 분포")
plt.show()

# 4. 데이터 스케일링

# 4-1. StandardScaler: 평균 0, 표준편차 1로 변환하는 방법 (표준화)
scaler_standard = StandardScaler()
# fit_transform을 통해 학습 및 변환 수행
data_standard_scaled = scaler_standard.fit_transform(df[numeric_cols])
# 결과를 DataFrame으로 변환 (원본 인덱스와 컬럼 유지)
df_standard_scaled = pd.DataFrame(data_standard_scaled, columns=numeric_cols, index=df.index)
df_standard_scaled.to_csv('data/Advertising_standard_Scaled.csv')
# 4-2. MinMaxScaler: 모든 값을 0~1 범위로 변환
scaler_minmax = MinMaxScaler()
data_minmax_scaled = scaler_minmax.fit_transform(df[numeric_cols])
df_minmax_scaled = pd.DataFrame(data_minmax_scaled, columns=numeric_cols, index=df.index)
df_minmax_scaled.to_csv('data/Advertising_minmax_Scaled.csv')
# 5. 스케일링 결과 시각화

# StandardScaler와 MinMaxScaler 적용 후 Box Plot 비교
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df_standard_scaled)
plt.title("StandardScaler 적용 후 데이터 분포")

plt.subplot(1, 2, 2)
sns.boxplot(data=df_minmax_scaled)
plt.title("MinMaxScaler 적용 후 데이터 분포")

plt.tight_layout()
plt.show()

# 6. 스케일링 결과 일부 출력하여 확인
print("\n[StandardScaler] 적용 후 데이터의 첫 5행:")
print(df_standard_scaled.head())

print("\n[MinMaxScaler] 적용 후 데이터의 첫 5행:")
print(df_minmax_scaled.head())