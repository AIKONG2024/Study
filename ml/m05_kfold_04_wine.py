import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

x,y = load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5 #데이터의 개수 까지 가능
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = ExtraTreesRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[kfold 적용 후]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

'''
============================================================
[The Best score] :  0.9603571428571429
[The Best model] :  ExtraTreesRegressor
============================================================
[kfold 적용 후]
acc :  [0.97784615 0.96790361 0.96177619 0.94802254 0.92548125]
평균 acc : 0.9562
============================================================
'''