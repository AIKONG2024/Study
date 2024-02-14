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

x = x.drop(
    [
        "hour_bef_windspeed",
        "hour_bef_pm2.5",
    ],
    axis=1,
)
test_csv = test_csv.drop(
    [
        "hour_bef_windspeed",
        "hour_bef_pm2.5",
    ],
    axis=1,
)

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
    # print(type(model).__name__ ,":", model.feature_importances_)
    
    # 25%이하 컬럼
    # feature_importance_set = pd.DataFrame({'feature': x.columns, 'importance':model.feature_importances_})
    # feature_importance_set.sort_values('importance', inplace=True)
    # delete_0_25_features = feature_importance_set['feature'][:int(len(feature_importance_set.values) * 0.25)]
    # delete_0_25_importance = feature_importance_set['importance'][:int(len(feature_importance_set.values) * 0.25)]
    # print(f'''
    # 제거 할 컬럼명 : 
    # {delete_0_25_features}  
    # 제거 할 feature_importances_ : 
    # {delete_0_25_importance}    
    # ''')

    y_submit = model.predict(test_csv) # count 값이 예측됨.
    submission_csv['count'] = y_submit

    ######### submission.csv 만들기(count컬럼에 값만 넣어주면됨) ############
    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}_" 
    file_path = path + f"submission_{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)

'''
[XGBRegressor] model r2 :  0.7574511135565787
[XGBRegressor] mode eval_r2 :  0.7574511135565787
[XGBRegressor] mode eval_mse :  1689.6014902447557
===================
제거후
[XGBRegressor] model r2 :  0.7574275535406655
[XGBRegressor] mode eval_r2 :  0.7574275535406655
[XGBRegressor] mode eval_mse :  1689.7656098932969
'''