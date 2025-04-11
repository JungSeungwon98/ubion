import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import platform
import seaborn as sns
# scikit-learn에서 데이터 스케일링을 위한 라이브러리 임포트
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 운영체제에 맞게 한글을 지원하는 폰트를 설정합니다.
if platform.system() == 'Windows':
    mpl.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    mpl.rcParams['font.family'] = 'AppleGothic'
else:
    mpl.rcParams['font.family'] = 'NanumGothic'

# 1. 데이터 불러오기
input_file = "data/Social_Network_Ads.csv"   # 원본 데이터 파일 경로
# 'Unnamed: 0' 컬럼은 인덱스로 지정하여 제거하고, 한글 인코딩 문제가 있으면 encoding 옵션 사용
df = pd.read_csv(input_file, index_col=0, encoding="utf-8")
print("원본 데이터 크기:", df.shape)
print("컬럼 목록:", df.columns.tolist())

# 2. 숫자형 컬럼만 선택 (광고 데이터셋은 보통 TV, Radio, Newspaper, Sales 등)
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 3. IQR 방법을 사용하여 이상치 제거
k = 1.5  # k 값 (보통 1.5 사용)
mask = pd.Series(True, index=df.index)

print("\n각 변수별 사분위수 및 경계값 계산:")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
    
    print(f"{col}: Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}, 하한 = {lower_bound:.2f}, 상한 = {upper_bound:.2f}")

print("\n정상 데이터 개수:", mask.sum())

# 마스크를 이용하여 이상치가 제거된 데이터프레임 생성
df_clean = df[mask].copy()

print("\n이상치 제거 전 데이터 크기:", df.shape)
print("이상치 제거 후 데이터 크기:", df_clean.shape)

# 4. 이상치 제거 전후의 분포를 시각화 (Box Plot)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df[numeric_cols])
plt.title("이상치 제거 전")

plt.subplot(1, 2, 2)
sns.boxplot(data=df_clean[numeric_cols])
plt.title("이상치 제거 후")

plt.tight_layout()
plt.show()

# 5. 자동으로 생성되는 출력 파일 이름: 원본 파일명 앞부분에 "_outlier_removed"를 붙여서 저장
# os.path 모듈을 이용하여 파일명과 확장자를 분리합니다.
dir_name = os.path.dirname(input_file)
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = os.path.join(dir_name, base_name + "_outlier_removed.csv")

df_clean.to_csv(output_file, encoding="utf-8")
print(f"\n이상치 제거된 데이터는 '{output_file}' 파일로 저장되었습니다.")