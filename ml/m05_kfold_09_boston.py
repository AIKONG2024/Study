import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

datasets= load_boston()
x = datasets.data
y = datasets.target

#데이터 전처리

n_splits = 5 
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
[The Best score] :  0.8612048082268853
[The Best model] :  ExtraTreesRegressor
============================================================
[kfold 적용 후]
acc :  [0.82350563 0.89434925 0.91715707 0.9321717  0.82515008]
평균 acc : 0.8785
============================================================
'''