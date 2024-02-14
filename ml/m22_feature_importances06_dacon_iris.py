#https://dacon.io/competitions/open/236070/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

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


random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]
for model in models:
    #컴파일 , 훈련
    model.fit(x_train, y_train)

    #평가, 예측
    acc = model.score(x_test, y_test)
    y_predict = model.predict(x_test)

    acc_pred = accuracy_score(y_test, y_predict) 
    submission = model.predict(test_csv)
    submission_csv['species'] = submission
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)
    print(type(model).__name__ ,":", model.feature_importances_)
'''
[DecisionTreeClassifier] model acc :  1.0
[DecisionTreeClassifier] eval_acc :  1.0
DecisionTreeClassifier : [0.         0.02381965 0.87386208 0.10231826]
[RandomForestClassifier] model acc :  1.0
[RandomForestClassifier] eval_acc :  1.0
RandomForestClassifier : [0.09937237 0.05000563 0.43570781 0.41491419]
[GradientBoostingClassifier] model acc :  1.0
[GradientBoostingClassifier] eval_acc :  1.0
GradientBoostingClassifier : [0.00335522 0.02353158 0.69599961 0.27711359]
[XGBClassifier] model acc :  0.9722222222222222
[XGBClassifier] eval_acc :  0.9722222222222222
XGBClassifier : [0.0102429  0.04045254 0.77684844 0.17245616]
'''
