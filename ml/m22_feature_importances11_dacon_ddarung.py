# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null

x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
y = train_csv['count'] 

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=12345)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)
#Early Stopping


random_state=42
#모델구성
models = [DecisionTreeRegressor(random_state=random_state), RandomForestRegressor(random_state=random_state),
          GradientBoostingRegressor(random_state=random_state), XGBRegressor(random_state=random_state)]
for model in models:

    #3. 컴파일, 훈련
    model.fit(x_train, y_train) 

    #4. 평가, 예측
    from sklearn.metrics import mean_squared_error
    r2 = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    r2_pred = r2_score(y_test, y_predict)
    mse_loss = mean_squared_error(y_test, y_predict)
    print(f"[{type(model).__name__}] model r2 : ", r2)
    print(f"[{type(model).__name__}] mode eval_r2 : ", r2_pred)
    print(f"[{type(model).__name__}] mode eval_mse : ", mse_loss)
    print(type(model).__name__ ,":", model.feature_importances_)

    y_submit = model.predict(test_csv) # count 값이 예측됨.
    submission_csv['count'] = y_submit

    ######### submission.csv 만들기(count컬럼에 값만 넣어주면됨) ############
    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}_" 
    file_path = path + f"submission_{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)

'''
[DecisionTreeRegressor] model r2 :  0.6252332021327116
[DecisionTreeRegressor] mode eval_r2 :  0.6252332021327116
[DecisionTreeRegressor] mode eval_mse :  2610.634703196347
DecisionTreeRegressor : [0.58739716 0.18435237 0.02816769 0.02372751 0.03822008 0.04545895
 0.03524901 0.03869397 0.01873325]
[RandomForestRegressor] model r2 :  0.76682403725006
[RandomForestRegressor] mode eval_r2 :  0.76682403725006
[RandomForestRegressor] mode eval_mse :  1624.3094739726027
RandomForestRegressor : [0.58208408 0.20584286 0.01811447 0.02813937 0.03512562 0.0333869
 0.03833832 0.03427195 0.02469642]
[GradientBoostingRegressor] model r2 :  0.752892496214042
[GradientBoostingRegressor] mode eval_r2 :  0.752892496214042
[GradientBoostingRegressor] mode eval_mse :  1721.35693042981
GradientBoostingRegressor : [0.64653522 0.22677103 0.01917036 0.00976463 0.01256082 0.02969372
 0.02902852 0.01901354 0.00746216]
[XGBRegressor] model r2 :  0.7574511135565787
[XGBRegressor] mode eval_r2 :  0.7574511135565787
[XGBRegressor] mode eval_mse :  1689.6014902447557
XGBRegressor : [0.31166038 0.09592723 0.44104397 0.01684011 0.02527023 0.02471927
 0.03169514 0.03320613 0.01963754]
'''