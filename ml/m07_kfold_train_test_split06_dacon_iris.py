#https://dacon.io/competitions/open/236070/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,cross_val_predict
from sklearn.utils import all_estimators
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

path = "C:/_data/dacon/iris/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

#데이터 확인
x = train_csv.drop(columns='species')
y = train_csv['species']

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
model = AdaBoostClassifier()

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
[The Best score] :  1.0
[The Best model] :  AdaBoostClassifier
============================================================
[kfold 적용 후]
acc :  [0.95833333 0.875      0.91666667 1.         0.95833333]
평균 acc : 0.9417
============================================================
[Stratifiedkfold 적용 후]
acc :  [0.95833333 0.91666667 0.91666667 0.75       0.95833333]
평균 acc : 0.9
============================================================
[스케일링 + train_test_split]
acc :  [0.95833333 0.91666667 0.91666667 0.75       0.95833333]
평균 acc : 0.9
eval acc_ : 0.9444444444444444
'''