import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

x,y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234, stratify=y)
print(x_test.shape)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5 #데이터의 개수 까지 가능
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = BaggingClassifier()

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
[The Best score] :  0.9593870479162842
[The Best model] :  BaggingClassifier
============================================================
[kfold 적용 후]
acc :  [0.96116279 0.96124024 0.96205745 0.96233283 0.96173904]
평균 acc : 0.9617
============================================================
[스케일링 + train_test_split]
acc :  [0.96210081 0.96048295 0.96229841 0.96263403 0.96097313]
평균 acc : 0.9617
'''