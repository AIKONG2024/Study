import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.datasets import load_diabetes
from sklearn.linear_model import HuberRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

n_splits = 5 #데이터의 개수 까지 가능
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = HuberRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[kfold 적용 후]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

'''
============================================================
[The Best score] :  0.6505589783375773
[The Best model] :  HuberRegressor
============================================================
[kfold 적용 후]
acc :  [0.57904986 0.41033047 0.51207251 0.59177537 0.29010322]
평균 acc : 0.4767
============================================================
[Stratifiedkfold 적용 후]
acc :  [0.44425976 0.59063908 0.43761107 0.46371011 0.51768554] 
평균 acc : 0.4908
'''