import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
)



import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = AdaBoostClassifier()

# 3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kf)

# 평가, 예측
print("[kfold 적용 후]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
# print(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print("eval acc_ :", acc_score) #ecal acc_ : 0.9

'''
============================================================
[The Best score] :  1.0
[The Best model] :  AdaBoostClassifier
============================================================
[kfold 적용 후]
acc :  [0.9        0.96666667 0.93333333 0.93333333 0.86666667]
평균 acc : 0.92
============================================================
[Stratifiedkfold 적용 후]
acc :  [0.96666667 1.         0.9        0.9        0.9       ]
평균 acc : 0.9333
============================================================
[스케일링 + train_test_split]
acc :  [0.91666667 0.91666667 1.         0.91666667 0.91666667]
평균 acc : 0.9333
eval acc_ : 0.9
'''
