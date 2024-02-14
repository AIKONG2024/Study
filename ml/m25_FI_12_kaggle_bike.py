import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error
path = 'C:/_data/kaggle/bike/'
train_csv =pd.read_csv(path + 'train.csv', index_col=0)
test_csv =pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

#데이터 전처리
x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
y = train_csv['count']

x = x.drop(
    [
        "windspeed",
        "holiday",
    ],
    axis=1,
)
test_csv = test_csv.drop(
    [
        "windspeed",
        "holiday",
    ],
    axis=1,
)
    
#데이터
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, random_state= 12345)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

random_state=42
#모델구성
models = [DecisionTreeRegressor(random_state=random_state), RandomForestRegressor(random_state=random_state),
          GradientBoostingRegressor(random_state=random_state), XGBRegressor(random_state=random_state)]
for model in models:
    # 컴파일, 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    r2 = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    r2_pred = r2_score(y_test, y_predict)
    mse_loss = mean_squared_error(y_test, y_predict)
    submit = model.predict(test_csv)
    
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
    
    # #데이터 출력
    submission_csv['count'] = submit
    import time as tm
    ltm = tm.localtime()
    file_name = f'sampleSubmission_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}.csv'
    submission_csv.to_csv(path + file_name, index = False )


'''
[XGBRegressor] model r2 :  0.32819435987379497
[XGBRegressor] mode eval_r2 :  0.32819435987379497
[XGBRegressor] mode eval_mse :  21714.856799231482
===================
제거후
[XGBRegressor] model r2 :  0.31618185350935857
[XGBRegressor] mode eval_r2 :  0.31618185350935857
[XGBRegressor] mode eval_mse :  22103.1385282366
'''