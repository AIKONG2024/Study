import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)


#모델 구성
model = LinearSVC(C=230)
#컴파일 , 훈련
model.fit(x_train, y_train)

#평가 예측
from sklearn.metrics import accuracy_score
loss = model.score(x_test, y_test)
x_predict = model.predict(x_test)
acc_score = accuracy_score(y_test, x_predict)

print("acc : ", loss)
print(f"accuracy_score : {acc_score}")
print("예측값 : ", x_predict)

'''
기존: accuracy : 0.9298245614035088
====================================
ML
acc :  0.9239766081871345
accuracy_score : 0.9239766081871345
'''