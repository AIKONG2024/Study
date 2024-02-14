import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)

random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]
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
    print(type(model).__name__ ,":", model.feature_importances_)

'''
[DecisionTreeClassifier] model acc :  0.935672514619883
[DecisionTreeClassifier] eval_acc :  0.935672514619883
DecisionTreeClassifier : [0.00811318 0.         0.00540878 0.         0.         0.
 0.         0.76884437 0.         0.         0.         0.
 0.         0.01040987 0.         0.         0.         0.00721171
 0.         0.         0.00256473 0.05316347 0.08837883 0.03053066
 0.00509145 0.         0.         0.         0.         0.02028294]
[RandomForestClassifier] model acc :  0.9298245614035088
[RandomForestClassifier] eval_acc :  0.9298245614035088
RandomForestClassifier : [0.05051396 0.01427253 0.06535006 0.05267564 0.00548866 0.01451934
 0.07444801 0.14079942 0.00457934 0.00228533 0.01148132 0.00305507
 0.00988702 0.02600549 0.0020226  0.00324998 0.00211034 0.00390562
 0.00247819 0.00321176 0.06986906 0.01550667 0.09150798 0.10358579
 0.01007552 0.01737333 0.03387804 0.1482521  0.00994496 0.00766688]
[GradientBoostingClassifier] model acc :  0.9239766081871345
[GradientBoostingClassifier] eval_acc :  0.9239766081871345
GradientBoostingClassifier : [1.88271033e-05 9.29662440e-03 1.17884272e-05 1.58662097e-05
 6.25522176e-04 1.16400328e-03 2.66560895e-03 3.19251773e-01
 1.18272801e-04 2.67851561e-04 2.37826010e-03 6.42121234e-04
 4.17493154e-03 2.07527395e-02 1.80036945e-06 3.18527361e-04
 2.30964669e-05 5.11482653e-04 1.38041658e-03 3.31665297e-03
 2.29579161e-02 2.84795564e-02 1.05062283e-01 4.83239878e-02
 1.20096918e-02 7.85741901e-05 3.46964830e-03 4.05823438e-01
 1.37527298e-03 5.48346379e-03]
[XGBClassifier] model acc :  0.9415204678362573
[XGBClassifier] eval_acc :  0.9415204678362573
XGBClassifier : [0.00675709 0.01160624 0.00232558 0.02561059 0.00553421 0.
 0.00733933 0.3879991  0.00441304 0.00074039 0.00501546 0.00106885
 0.         0.00645363 0.0024287  0.01074255 0.         0.00575032
 0.00267557 0.00300085 0.06041276 0.02622655 0.08620969 0.03637366
 0.0086379  0.         0.00983377 0.27401802 0.00312537 0.00570074]
'''