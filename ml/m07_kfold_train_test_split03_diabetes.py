import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, cross_val_predict
from sklearn.datasets import load_diabetes
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8)

n_splits = 5 #데이터의 개수 까지 가능
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = HuberRegressor()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kf)

# 평가, 예측
print("[kfold 적용 후]")
print("r2 : ", scores, "\n평균 r2 :", round(np.mean(scores),4))
y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
# print(y_predict)
score = r2_score(y_test, y_predict)
print("eval r2_ :", score) #ecal acc_ : 0.9

'''
============================================================
[The Best score] :  0.6505589783375773
[The Best model] :  HuberRegressor
============================================================
[kfold 적용 후]
r2 :  [0.57904986 0.41033047 0.51207251 0.59177537 0.29010322]
평균 r2 : 0.4767
============================================================
[Stratifiedkfold 적용 후]
r2 :  [0.44425976 0.59063908 0.43761107 0.46371011 0.51768554] 
평균 42 : 0.4908

eval r2_ : 0.5140983839395773
'''