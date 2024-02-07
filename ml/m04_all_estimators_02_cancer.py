import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)

allAlgorithms = all_estimators(type_filter='classifier') #41개
# allAlgorithms = all_estimators(type_filter='regressor') #55개

# 모델구성
for name, algorithm in allAlgorithms :
    try:
        # 모델
        model = algorithm()
        # 훈련
        model.fit(x_train, y_train)

        # 평가, 예측
        results = model.score(x_test, y_test)
        print(f"[{name}] score : ", results)
        x_predict = model.predict(x_test)
    except:
        continue

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