import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 데이터 로드 및 전처리
# CSV 파일 읽기 (파일 경로를 알맞게 수정하세요)
data = pd.read_csv('data/Country-data.csv')
print("원본 데이터 미리보기:")
print(data.head())

# 숫자형 변수만 선택 (예: 국가명이 포함된 컬럼은 제외)
numeric_data = data.select_dtypes(include=[np.number])
print("\n숫자형 데이터 컬럼:")
print(numeric_data.columns)

# 만약 'Country'와 같이 국가명을 나타내는 컬럼이 있다면 따로 보관
if 'Country' in data.columns:
    countries = data['Country']
else:
    countries = None

# 2. 데이터 스케일링
scaler = StandardScaler()
data_scaled = scaler.fit_transform(numeric_data)

# 3. PCA 수행 (모든 주성분 계산)
pca = PCA()
pca_result = pca.fit_transform(data_scaled)

# 각 주성분의 설명 분산 비율 출력
explained_variance = pca.explained_variance_ratio_
print("\n각 주성분의 설명 분산 비율:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f}")

# 누적 설명 분산 계산
cumsum_variance = np.cumsum(explained_variance)

# 4. 시각화

# (1) 누적 설명 분산 플롯 (Cumulative Variance)
plt.figure(figsize=(8,6))
plt.plot(range(1, len(cumsum_variance)+1), cumsum_variance, marker='o', linestyle='--')
plt.title("누적 설명 분산")
plt.xlabel("주성분 개수")
plt.ylabel("누적 설명 분산 비율")
plt.grid(True)
plt.show()

# (2) Scree Plot: 각 주성분의 설명 분산 비율
plt.figure(figsize=(8,6))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, align='center')
plt.xlabel("주성분")
plt.ylabel("설명 분산 비율")
plt.title("Scree Plot")
plt.show()

# (3) biplot 함수 정의
def biplot(scores, loadings, labels=None):
    """
    scores : PCA 변환된 점수 행렬 (샘플수 x 주성분 수)
    loadings: 각 변수의 로딩 (주성분 축에 대한 계수)
    labels: 각 샘플의 레이블 (예: 국가명)
    """
    xs = scores[:, 0]
    ys = scores[:, 1]
    plt.figure(figsize=(10,8))
    plt.scatter(xs, ys, alpha=0.7)
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (xs[i], ys[i]), fontsize=8, alpha=0.75)
    # 변수(특성) 벡터 표시 (첫 두 주성분 기준)
    for i in range(loadings.shape[0]):
        plt.arrow(0, 0, loadings[i, 0]*max(xs), loadings[i, 1]*max(ys),
                  color='r', width=0.005, head_width=0.05)
        plt.text(loadings[i, 0]*max(xs)*1.1, loadings[i, 1]*max(ys)*1.1, 
                 numeric_data.columns[i], color='r', fontsize=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biplot (PC1 vs PC2)")
    plt.grid(True)
    plt.show()

# 첫 두 주성분에 대한 로딩(계수)
loadings = pca.components_.T  # 각 행은 원래 변수, 각 열은 주성분
print("\n각 변수의 PC1, PC2 로딩:")
loading_df = pd.DataFrame(loadings[:, :2], index=numeric_data.columns, columns=['PC1', 'PC2'])
print(loading_df)

# biplot 그리기 (만약 국가명이 존재하면 label로 사용)
biplot(pca_result, loadings, labels=countries)

# 5. PCA 결과를 데이터프레임에 저장
# 각 샘플의 주성분 점수를 컬럼에 저장하고, 국가명(존재 시)도 함께 저장
pca_columns = [f"PC{i+1}" for i in range(pca_result.shape[1])]
pca_df = pd.DataFrame(pca_result, columns=pca_columns)
if countries is not None:
    pca_df.insert(0, "Country", countries)
    
# PCA 결과 저장 (CSV 파일)
pca_df.to_csv("Country-data_PCA.csv", index=False)
print("\nPCA 결과가 'Country-data_PCA.csv' 파일로 저장되었습니다.")