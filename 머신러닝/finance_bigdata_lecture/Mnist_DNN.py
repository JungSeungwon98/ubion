import time
start_time = time.time()  # 전체 실행 시작 시간 측정

# 필요한 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 1. MNIST 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 학습 전 이미지 몇 개 출력하기
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3. 데이터 전처리
# 이미지는 DNN에 사용할 수 있도록 1차원으로 펼쳐줍니다.
# 그리고 0~255 범위의 픽셀 값을 0~1 사이로 정규화합니다.
x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255.0

# 레이블은 one-hot encoding 처리 (클래스가 10개)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. DNN 모델 구성 (CNN 없이 완전 연결 층만 사용)
model = Sequential()
model.add(Dense(512, input_shape=(28*28,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10개의 클래스
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 5. 모델 학습
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# 6. 테스트 데이터로 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# 7. 테스트 이미지 중 하나의 예측 결과 확인
predictions = model.predict(x_test)
predicted_label = np.argmax(predictions[0])
true_label = np.argmax(y_test[0])
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"True: {true_label}, Predicted: {predicted_label}")
plt.axis('off')
plt.show()

end_time = time.time()  # 전체 실행 종료 시간 측정
elapsed_time = end_time - start_time
print(f"전체 실행 시간: {elapsed_time:.2f}초")