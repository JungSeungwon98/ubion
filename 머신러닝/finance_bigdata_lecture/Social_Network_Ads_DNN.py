# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 데이터셋 불러오기 (CSV 파일은 현재 작업 디렉토리에 있어야 합니다.)
data = pd.read_csv('data/Social_Network_Ads.csv')

# 데이터의 처음 몇 줄을 출력해 확인
print(data.head())

# 데이터셋 컬럼: 'User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased'
# 여기서는 'Age'와 'EstimatedSalary'를 특성으로 사용합니다.
X = data[['Age', 'EstimatedSalary']].values

# 레이블(y)은 'Purchased' 컬럼에서 가져옵니다.
y = data['Purchased'].values

# 데이터를 훈련셋과 테스트셋으로 분할 (훈련: 80%, 테스트: 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 특성 스케일링: StandardScaler 사용
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 딥 뉴럴 네트워크(DNN) 모델 구축
model = Sequential([
    # 첫 번째 은닉층: 32개의 뉴런과 ReLU 활성화 함수
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    
    # 두 번째 은닉층: 16개의 뉴런과 ReLU 활성화 함수
    Dense(16, activation='relu'),
    Dropout(0.2),
    
    # 출력층: 1개의 뉴런, sigmoid 활성화 함수 (이진 분류일때는 시그모이드 사용)
    Dense(1, activation='sigmoid')
])

# 모델 컴파일 (학습시키기 위한 준비단계): Adam 옵티마이저 및 이진 교차 엔트로피 손실 함수 사용
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 훈련: 100 에포크, 배치 사이즈 10, 검증 데이터는 X_test와 y_test 사용
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

# 테스트 데이터로 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"테스트 손실: {loss}")
print(f"테스트 정확도: {accuracy}")

# 모델 파일명을 생성하고 학습된 모델 저장
model_filename = "구매예측_DNN.h5"
model.save(model_filename)
print(f"모델이 '{model_filename}' 이름으로 저장되었습니다.")