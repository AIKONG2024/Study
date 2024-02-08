import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold ,cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_breast_cancer(return_X_y=True)
n_splits = 5 #데이터의 개수 까지 가능
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = LinearDiscriminantAnalysis()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[kfold 적용 후]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

'''
============================================================
[The Best score] :  0.9415204678362573
[The Best model] :  LinearDiscriminantAnalysis
============================================================
[kfold 적용 후]
acc :  [0.97368421 0.96491228 0.93859649 0.95614035 0.9380531 ]
평균 acc : 0.9543
============================================================
'''