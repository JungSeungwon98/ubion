# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# 1. 데이터 불러오기
# 파일명이 'Social_Network_Ads.csv'라고 가정합니다.
df = pd.read_csv('data/Social_Network_Ads.csv')

# 2. 데이터 기본 정보 확인
print("==== 데이터의 처음 5행 ====")
print(df.head(), "\n")

print("==== 데이터의 요약 정보 (info) ====")
df.info()
print("\n==== 기술 통계 (describe) ====")
print(df.describe(), "\n")

# 3. 결측치 확인 (각 컬럼별 결측값의 개수)
print("==== 결측치 확인 ====")
print(df.isnull().sum(), "\n")

# 4. 각 변수의 분포 시각화

# 4-1. Age (연령) 변수 분포: 히스토그램
plt.figure()  # 새로운 그림 생성
plt.hist(df['Age'], bins=20)
plt.title('Age 분포')
plt.xlabel('Age')
plt.ylabel('빈도수')
plt.show()

# 4-2. EstimatedSalary (예상 연봉) 변수 분포: 히스토그램
plt.figure()
plt.hist(df['EstimatedSalary'], bins=20)
plt.title('EstimatedSalary 분포')
plt.xlabel('EstimatedSalary')
plt.ylabel('빈도수')
plt.show()

# 4-3. Purchased (구매 여부) 변수 분포: 막대그래프
purchased_counts = df['Purchased'].value_counts()
plt.figure()
plt.bar(purchased_counts.index.astype(str), purchased_counts.values)
plt.title('Purchased 분포')
plt.xlabel('Purchased')
plt.ylabel('Count')
plt.show()

# 4-4. Gender (성별) 변수 분포: 막대그래프
if 'Gender' in df.columns:
    gender_counts = df['Gender'].value_counts()
    plt.figure()
    plt.bar(gender_counts.index, gender_counts.values)
    plt.title('Gender 분포')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

# 5. 상자그림(Box Plot)으로 이상치 및 분포 시각화

# 5-1. Age 상자 그림
plt.figure()
plt.boxplot(df['Age'])
plt.title('Age 상자 그림')
plt.ylabel('Age')
plt.show()

# 5-2. EstimatedSalary 상자 그림
plt.figure()
plt.boxplot(df['EstimatedSalary'])
plt.title('EstimatedSalary 상자 그림')
plt.ylabel('EstimatedSalary')
plt.show()

# 6. 산점도(Scatter Plot): Age와 EstimatedSalary 간의 관계를 Purchased(구매여부)별로 시각화
plt.figure()
for value in df['Purchased'].unique():
    subset = df[df['Purchased'] == value]
    plt.scatter(subset['Age'], subset['EstimatedSalary'], label=f'Purchased = {value}')
plt.title('Age와 EstimatedSalary의 산점도 (구매 여부별)')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.legend()
plt.show()

# 7. 변수 간 상관관계 분석: 상관계수 히트맵 (matplotlib의 imshow 사용)
corr = df.corr()
plt.figure()
plt.imshow(corr, cmap='viridis', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('변수 간 상관관계')
plt.show()

# 추가: 정규분포 상태를 체크하기 위한 왜도와 첨도 분석
# 여기서는 Age와 EstimatedSalary 변수에 대해 왜도와 첨도를 계산합니다.
print("==== 정규분포 체크: 왜도와 첨도 분석 ====")
numerical_features = ['Age', 'EstimatedSalary']
for feature in numerical_features:
    skewness = df[feature].skew()
    kurtosis = df[feature].kurtosis()  # 기본적으로 피어슨 첨도(Pearson's Kurtosis) 계산
    print(f"{feature}의 왜도(비대칭도): {skewness:.2f}")
    print(f"{feature}의 첨도(꼬리두께): {kurtosis:.2f}\n")