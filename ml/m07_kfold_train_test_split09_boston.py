import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

datasets= load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5 
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = ExtraTreesRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[스케일링 + train_test_split]")
print("r2 : ", scores, "\n평균 acc :", round(np.mean(scores),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
# print(y_predict)
acc_score = r2_score(y_test, y_predict)
print("eval r2 :", acc_score) 

'''
============================================================
[The Best score] :  0.8612048082268853
[The Best model] :  ExtraTreesRegressor
============================================================
[kfold 적용 후]
acc :  [0.82350563 0.89434925 0.91715707 0.9321717  0.82515008]
평균 acc : 0.8785
============================================================
[스케일링 + train_test_split]
r2 :  [0.84549076 0.8915841  0.91118302 0.93238538 0.82918008]
평균 acc : 0.882
eval r2 : 0.8292094395777979
'''