from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, random_state=2874458)

# 모델 구성
from sklearn.svm import LinearSVR
model = LinearSVR()

# 컴파일, 훈련
model.fit(x_train, y_train)

# 평가, 예측
loss = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
result = model.predict(x)
print("r2 : ", loss)
print("r2_score : ", r2)


# loss :  0.5457435846328735
# R2 :  0.5076286753020893
# 걸린시간 :  138.48 초

# loss :  0.4252395756878554
# R2 :  0.4252395756878554
