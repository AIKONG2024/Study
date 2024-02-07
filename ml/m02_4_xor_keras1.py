import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0]) 
print(x_data.shape, y_data.shape)#(4, 2) (4,)


# 2. 모델 구성
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(1,input_dim = 2, activation='sigmoid'))


# 3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
# acc = model.score(x_data, y_data)
result = model.evaluate(x_data, y_data)
# print("model.score:", acc)
print("acc:", result[1])

y_predict = np.round(model.predict(x_data)).reshape(-1,).astype(int)
acc2 = accuracy_score(y_data, y_predict)
print("accuracy_score:", acc2)

print("===========================")
print(y_data)
print(y_predict)
