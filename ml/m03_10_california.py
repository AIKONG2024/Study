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
model = LinearSVR(C=300)

# 컴파일, 훈련
model.fit(x_train, y_train)

# 평가, 예측
r2 = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2_pred = r2_score(y_test, y_predict)
result = model.predict(x)
print("r2 : ", r2)
print("eval_r2 : ", r2_pred)

'''
r2 :  -0.17073552440929052
r2_score :  -0.17073552440929052
'''

