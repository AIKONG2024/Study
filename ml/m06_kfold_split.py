import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_iris()
df = pd.DataFrame(datasets.data, columns = datasets.feature_names) # x,columns 넣음

# print(df) #[150 rows x 4 columns]

n_splits = 3 #데이터의 개수 까지 가능
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# split -> df의 잘린 부분이 for 문 돌면서 train_index, val_index에 들어감
for train_index, val_index in kf.split(df):
    print("===============================")
    print(train_index, '\n', val_index)
    print(
        f"""
          훈련 데이터 개수 : {len(train_index)}
          검증 데이터 개수 : {len(val_index)}
          """
    )

# 모델구성
model = AdaBoostClassifier()

# 3. 훈련
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
[Stratifiedkfold 적용 후]
acc :  [0.96666667 1.         0.9        0.9        0.9       ]
평균 acc : 0.9333
'''
