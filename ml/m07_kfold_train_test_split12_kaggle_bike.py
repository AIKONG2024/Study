import pandas as pd
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = 'C:/_data/kaggle/bike/'
train_csv =pd.read_csv(path + 'train.csv', index_col=0)
test_csv =pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

#데이터 전처리
x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
    
#데이터
n_splits = 5 
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = HistGradientBoostingRegressor()

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
[The Best score] :  0.35691020073346846
[The Best model] :  HistGradientBoostingRegressor      
============================================================
[kfold 적용 후]
acc :  [0.33787554 0.34673456 0.35355217 0.38048027 0.37563452]
평균 acc : 0.358
============================================================
[스케일링 + train_test_split]
r2 :  [0.33787554 0.34673456 0.35355217 0.38048027 0.37563452]
평균 acc : 0.3589
eval r2 : 0.29744093117836146
'''