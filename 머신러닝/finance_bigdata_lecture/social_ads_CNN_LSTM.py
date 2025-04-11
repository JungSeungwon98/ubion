# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import platform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout

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
data = pd.read_csv('data/Social_Network_Ads.csv')
print("데이터 미리보기:")
print(data.head())

# 2. 특성과 레이블 선정
# 일반적으로 Social_Network_Ads.csv 파일은 ['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased'] 컬럼을 가집니다.
# 여기서는 'Age'와 'EstimatedSalary'를 특성으로, 'Purchased'를 레이블로 사용합니다.
X = data[['Age', 'EstimatedSalary']].values    # shape: (샘플수, 2)
y = data['Purchased'].values                   # 0 또는 1

# 3. 데이터 전처리
# 특성 스케일링 (표준화)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 각 특성의 값이 평균 0, 분산 1이 되도록 변환

# CNN+LSTM 모델을 위한 시퀀스 데이터로 reshape
# 여기서는 각 샘플(2개의 특성)을 sequence length=2, feature dimension=1인 형태로 변환합니다.
# 최종 shape: (샘플수, 2, 1)
X_seq = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# 4. 데이터 분할: 훈련셋과 테스트셋 (전체의 80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=0)

# 5. CNN + LSTM 하이브리드 모델 구성
model = Sequential()
# 1차원 합성곱 레이어: 필터 32, kernel size 1 (입력 shape: (2, 1))
model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(2, 1)))
model.add(Dropout(0.2))
# 추가적인 1차원 합성곱 레이어로 특징 추출
model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
# LSTM 레이어: 시퀀스 특성을 학습 (유닛 32 사용)
model.add(LSTM(32, activation='tanh'))
model.add(Dropout(0.2))
# 출력층: 이진 분류를 위한 sigmoid 활성화 (구매: 1, 미구매: 0)
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 6. 모델 학습
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# 7. 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"테스트 손실: {loss:.4f}")
print(f"테스트 정확도: {accuracy:.4f}")

# 8. 예측 예제: 테스트 데이터의 첫 번째 샘플에 대한 예측
sample_prediction = model.predict(X_test[0:1])
predicted_class = 1 if sample_prediction[0][0] >= 0.5 else 0
print(f"예측 확률: {sample_prediction[0][0]:.4f}, 예측 클래스: {predicted_class}")

# (선택사항) 학습 과정 시각화
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title("훈련 및 검증 정확도")
plt.xlabel("에포크")
plt.ylabel("정확도")
plt.legend()
plt.show()