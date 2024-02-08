import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

x,y = load_wine(return_X_y=True)
x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5 #데이터의 개수 까지 가능
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
n_splits = 5 #데이터의 개수 까지 가능
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = AdaBoostClassifier()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kf)

# 평가, 예측
print("[스케일링 + train_test_split]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
# print(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print("eval acc_ :", acc_score) #ecal acc_ : 0.9

'''
============================================================
[The Best score] :  0.9603571428571429
[The Best model] :  ExtraTreesRegressor
============================================================
[kfold 적용 후]
acc :  [0.97784615 0.96790361 0.96177619 0.94802254 0.92548125]
평균 acc : 0.9562
============================================================
[Stratifiedkfold 적용 후]
acc :  [0.92851624 0.95609645 0.94790051 0.98092355 0.95336351] 
평균 acc : 0.9534
============================================================
[스케일링 + train_test_split]
acc :  [0.82758621 0.93103448 0.85714286 0.85714286 0.78571429]
평균 acc : 0.8517
eval acc_ : 0.9166666666666666
'''