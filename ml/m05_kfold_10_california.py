import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = fetch_california_housing(return_X_y=True)

# 데이터
n_splits = 5 
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = GradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[kfold 적용 후]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

'''
============================================================
[The Best score] :  0.7894213579770931
[The Best model] :  GradientBoostingRegressor
============================================================
[kfold 적용 후]
acc :  [0.79783784 0.8002211  0.78592059 0.76960237 0.77564235]
평균 acc : 0.7858
============================================================
'''