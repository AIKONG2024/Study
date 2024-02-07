import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)

models = [LinearSVC(), Perceptron(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models :
    #모델 구성
    
    #컴파일 , 훈련
    model.fit(x_train, y_train)

    #평가 예측
    from sklearn.metrics import accuracy_score
    acc = model.score(x_test, y_test)
    x_predict = model.predict(x_test)
    acc_pred = accuracy_score(y_test, x_predict)

    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)

'''
[LinearSVC] model acc :  0.9005847953216374
[LinearSVC] eval_acc :  0.9005847953216374
[Perceptron] model acc :  0.8304093567251462
[Perceptron] eval_acc :  0.8304093567251462
[LogisticRegression] model acc :  0.9181286549707602
[LogisticRegression] eval_acc :  0.9181286549707602
[KNeighborsClassifier] model acc :  0.935672514619883
[KNeighborsClassifier] eval_acc :  0.935672514619883
[DecisionTreeClassifier] model acc :  0.9239766081871345
[DecisionTreeClassifier] eval_acc :  0.9239766081871345
[RandomForestClassifier] model acc :  0.9181286549707602
[RandomForestClassifier] eval_acc :  0.9181286549707602
'''