from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.72, random_state=335688) 

# 모델 구성
from sklearn.svm import LinearSVC
model = LinearSVC(C=1)
model.fit(x_train, y_train)

# 컴파일 훈련

# 평가 예측
loss = model.score(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("r2 = ", r2)

'''
기존
로스 :  2372.69677734375
r2 =  0.6415152545466019
============================
ML
로스 :  0.008064516129032258
r2 =  0.2672879418024767
'''
