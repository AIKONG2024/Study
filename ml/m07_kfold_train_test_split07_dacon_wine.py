# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234, stratify=y)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#데이터 분류
n_splits = 5 
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = ExtraTreesClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[스케일링 + train_test_split]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
# print(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print("eval acc_ :", acc_score) #ecal acc_ : 0.9

'''
============================================================
[The Best score] :  0.6472727272727272
[The Best model] :  ExtraTreesClassifier
============================================================
[kfold 적용 후]
acc :  [0.66090909 0.68272727 0.67242948 0.67879891 0.67879891]
평균 acc : 0.6747
============================================================
[Stratifiedkfold 적용 후]
acc :  [0.67272727 0.69818182 0.6933576  0.65696087 0.68971793]
평균 acc : 0.6822
============================================================
[스케일링 + train_test_split]
acc :  [0.66727273 0.68272727 0.70154686 0.65059145 0.68152866] 
평균 acc : 0.6767
eval acc_ : 0.5866666666666667
'''