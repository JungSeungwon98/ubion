# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
data = pd.read_csv('data/Social_Network_Ads.csv')
print("데이터 미리보기:")
print(data.head())

# 2. 데이터 전처리
# 여기서는 'Age'와 'EstimatedSalary' 2개의 특성을 사용하며,
# 레이블은 'Purchased' 컬럼을 사용합니다.
X = data[['Age', 'EstimatedSalary']].values
y = data['Purchased'].values

# 데이터를 훈련셋과 테스트셋으로 분리 (훈련: 80%, 테스트: 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 특성 스케일링 (표준화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LSTM/GRU 모델은 3차원 입력을 필요로 합니다.
# 각 샘플을 시퀀스 길이 1로 변환 (형태: [샘플수, 타임스텝, 특성수])
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 3. 모델 생성 함수 정의

def create_lstm_model(input_shape):
    """LSTM 모델 생성 및 컴파일"""
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(input_shape):
    """GRU 모델 생성 및 컴파일"""
    model = Sequential()
    model.add(GRU(32, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 입력 데이터 형태 (타임스텝, 특성수)
input_shape = (X_train.shape[1], X_train.shape[2])

# 4. LSTM 모델 학습
print("\n===== LSTM 모델 학습 =====")
lstm_model = create_lstm_model(input_shape)
lstm_model.summary()

history_lstm = lstm_model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

# LSTM 모델 평가
loss_lstm, acc_lstm = lstm_model.evaluate(X_test, y_test)
print(f"LSTM 모델 - 테스트 손실: {loss_lstm:.4f}, 테스트 정확도: {acc_lstm:.4f}")

# LSTM 모델 저장
lstm_filename = "구매예측_LSTM.h5"
lstm_model.save(lstm_filename)
print(f"LSTM 모델이 '{lstm_filename}' 파일로 저장되었습니다.")

# 5. GRU 모델 학습
print("\n===== GRU 모델 학습 =====")
gru_model = create_gru_model(input_shape)
gru_model.summary()

history_gru = gru_model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

# GRU 모델 평가
loss_gru, acc_gru = gru_model.evaluate(X_test, y_test)
print(f"GRU 모델 - 테스트 손실: {loss_gru:.4f}, 테스트 정확도: {acc_gru:.4f}")

# GRU 모델 저장
gru_filename = "구매예측_GRU.h5"
gru_model.save(gru_filename)
print(f"GRU 모델이 '{gru_filename}' 파일로 저장되었습니다.")

# 6. 모델 성능 비교
print("\n===== 모델 성능 비교 =====")
print(f"LSTM 모델 - 손실: {loss_lstm:.4f}, 정확도: {acc_lstm:.4f}")
print(f"GRU 모델  - 손실: {loss_gru:.4f}, 정확도: {acc_gru:.4f}")

# 7. 새로운 데이터(예: 나이 22세, 연봉 30000)에 대한 예측 예제 (LSTM 모델 사용)
new_sample = np.array([[22, 30000]])
new_sample_scaled = scaler.transform(new_sample)
new_sample_scaled = new_sample_scaled.reshape((new_sample_scaled.shape[0], 1, new_sample_scaled.shape[1]))

# LSTM 모델을 이용한 예측
lstm_prediction = lstm_model.predict(new_sample_scaled)
lstm_prob = lstm_prediction[0][0]
print(f"\n[예측 예제] 나이 22, 연봉 30000인 사람의 LSTM 모델 구매 예측 확률: {lstm_prob:.4f}")
if lstm_prob >= 0.5:
    print("예측 결과: 구매(O)")
else:
    print("예측 결과: 미구매(X)")