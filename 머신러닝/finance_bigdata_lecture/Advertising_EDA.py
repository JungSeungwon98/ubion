# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import platform

# 파일 경로에 있는 CSV 파일을 읽어옵니다.
df = pd.read_csv("data/Advertising.csv", index_col=0)

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

# 음수 부호가 깨지지 않도록 설정
mpl.rcParams['axes.unicode_minus'] = False

# 데이터셋의 첫 5행을 출력하여 데이터 구조를 확인합니다.
print("데이터의 첫 5행:")
print(df.head())

# 데이터셋의 기초 통계량(평균, 표준편차, 최소/최대 값 등)을 확인합니다.
print("\n요약 통계량:")
print(df.describe())

# 데이터셋의 정보를 확인합니다. (컬럼명, 자료형, 결측치 유무 등)
print("\n데이터 정보:")
print(df.info())

# 각 컬럼별 결측치(누락된 값)가 있는지 확인합니다.
print("\n결측치 개수:")
print(df.isnull().sum())

# 변수 간 상관관계 행렬을 계산하여 출력합니다.
corr_matrix = df.corr()
print("\n상관관계 행렬:")
print(corr_matrix)

# 상관관계 행렬을 히트맵으로 시각화합니다.
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("변수 간 상관관계 히트맵")
plt.show()

# 각 변수의 분포를 히스토그램으로 확인합니다.
df.hist(bins=20, figsize=(12,10), color="skyblue", edgecolor="black")
plt.suptitle("각 변수의 분포 (히스토그램)")
plt.show()

# 변수 간 관계를 한눈에 보기 위해 Pair Plot(산점도 행렬)을 생성합니다.
sns.pairplot(df)
plt.suptitle("변수 간 산점도 행렬", y=1.02)
plt.show()

# Box Plot을 사용하여 각 변수의 분포와 이상치(outlier)를 확인합니다.
plt.figure(figsize=(12,8))
sns.boxplot(data=df, orient="h")
plt.title("각 변수의 Box Plot")
plt.show()

# 'Sales'와 다른 광고 채널(TV, Radio, Newspaper) 간의 관계를 산점도로 각각 시각화합니다.
features = ["TV", "Radio", "Newspaper"]
for feature in features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[feature], y=df["Sales"])
    plt.title(f"{feature} vs Sales")
    plt.xlabel(feature)
    plt.ylabel("Sales")
    plt.show()