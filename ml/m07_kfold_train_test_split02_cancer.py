import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold ,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5 #데이터의 개수 까지 가능
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = LinearDiscriminantAnalysis()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kf)

# 평가, 예측
print("[kfold 적용 후]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
# print(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print("eval acc_ :", acc_score) #ecal acc_ : 0.9

'''
============================================================
[The Best score] :  0.9415204678362573
[The Best model] :  LinearDiscriminantAnalysis
============================================================
[kfold 적용 후]
acc :  [0.97368421 0.96491228 0.93859649 0.95614035 0.9380531 ]
평균 acc : 0.9543
============================================================
[Stratifiedkfold 적용 후]
acc :  [0.93859649 0.92105263 0.98245614 0.98245614 0.94690265] 
평균 acc : 0.9543
eval acc_ : 0.9298245614035088
============================================================
[스케일링 + train_test_split]
acc :  [0.91666667 0.91666667 1.         0.91666667 0.91666667]
평균 acc : 0.9333
eval acc_ : 0.9
'''