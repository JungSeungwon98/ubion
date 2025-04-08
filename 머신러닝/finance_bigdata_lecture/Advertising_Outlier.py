import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import matplotlib as mpl

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
# 'Unnamed: 0' 컬럼은 인덱스로 지정하여 제거하고, 한글 인코딩 문제가 있으면 encoding 옵션 사용
df = pd.read_csv("data/Advertising.csv", index_col=0, encoding="utf-8")
print("원본 데이터 크기:", df.shape)
print("컬럼 목록:", df.columns.tolist())

# 2. 숫자형 컬럼만 선택 (광고 데이터셋은 보통 TV, Radio, Newspaper, Sales 등)
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 3. IQR 방법을 사용하여 이상치 제거
# k 값 (보통 1.5 사용)
k = 1.5

# 모든 숫자형 컬럼에 대해 이상치 제거 조건을 적용할 마스크 생성
mask = pd.Series(True, index=df.index)

print("\n각 변수별 사분위수 및 경계값 계산:")
for col in numeric_cols:
    # 사분위수 계산
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    # 해당 컬럼에 대해 정상범위에 있는 행을 True로 체크하여 마스크 업데이트
    mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
    
    print(f"{col}: Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}, 하한 = {lower_bound:.2f}, 상한 = {upper_bound:.2f}")

# 마스크를 이용하여 이상치가 제거된 데이터프레임 생성
df_clean = df[mask].copy()
df_clean.to_csv('outlier_removed.csv')

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