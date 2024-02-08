import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_iris(return_X_y=True)
n_splits = 5 #데이터의 개수 까지 가능
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = AdaBoostClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[kfold 적용 후]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

'''
============================================================
[The Best score] :  1.0
[The Best model] :  AdaBoostClassifier
============================================================
[kfold 적용 후]
acc :  [0.9        0.96666667 0.93333333 0.93333333 0.86666667]
평균 acc : 0.92
============================================================
'''