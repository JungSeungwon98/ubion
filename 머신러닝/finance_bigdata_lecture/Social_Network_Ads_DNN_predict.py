# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 저장된 모델 로드하기
model = tf.keras.models.load_model("구매예측_DNN.h5")
print("모델이 성공적으로 로드되었습니다.")

# 새로운 샘플: 나이 22세, 연봉 30000
new_person = np.array([[22, 30000]])

# 원래 학습 시 사용한 CSV 파일을 불러와서 scaler를 재생성
data = pd.read_csv('data/Social_Network_Ads.csv')
X = data[['Age', 'EstimatedSalary']].values

scaler = StandardScaler()
scaler.fit(X)  # 학습 데이터를 기반으로 스케일러를 피팅

# 새로운 샘플 데이터 스케일링
new_person_scaled = scaler.transform(new_person)

# 예측 수행 (출력은 확률로 반환됩니다.)
prediction = model.predict(new_person_scaled)
predicted_probability = prediction[0][0]
print(f"나이 22, 연봉 30000인 사람의 구매 예측 확률: {predicted_probability:.4f}")

# 일반적으로 0.5를 기준으로 이진 분류 결정을 내립니다.
if predicted_probability >= 0.5:
    print("예측 결과: 구매(O)")
else:
    print("예측 결과: 미구매(X)")