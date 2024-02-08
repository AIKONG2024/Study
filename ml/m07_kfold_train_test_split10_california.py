import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score,cross_val_predict
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 데이터
n_splits = 5 
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = GradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[스케일링 + train_test_split]")
print("r2 : ", scores, "\n평균 acc :", round(np.mean(scores),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
# print(y_predict)
acc_score = r2_score(y_test, y_predict)
print("eval r2 :", acc_score) 

'''
============================================================
[The Best score] :  0.7894213579770931
[The Best model] :  GradientBoostingRegressor
============================================================
[kfold 적용 후]
acc :  [0.79783784 0.8002211  0.78592059 0.76960237 0.77564235]
평균 acc : 0.7858
============================================================
[스케일링 + train_test_split]
r2 :  [0.79783784 0.80020632 0.78578784 0.76954398 0.77564449]
평균 acc : 0.7858
'''