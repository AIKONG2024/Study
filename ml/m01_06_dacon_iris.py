#https://dacon.io/competitions/open/236070/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = "C:/_data/dacon/iris/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

#데이터 확인
x = train_csv.drop(columns='species')
y = train_csv['species']

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=200, stratify=y)

#모델 생성
from sklearn.svm import LinearSVC
model = LinearSVC(C=100)

#컴파일 , 훈련
model.fit(x_train, y_train)

#평가, 예측
acc = model.score(x_test, y_test)
y_predict = model.predict(x_test)

acc_pred = accuracy_score(y_test, y_predict) 
submission = model.predict(test_csv)
submission_csv['species'] = submission
print("acc : ", acc)
print("eval_acc : ", acc_pred)
'''
acc :  0.9444444444444444
eval_acc :  0.9444444444444444
'''
