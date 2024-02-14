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
    print(type(model).__name__ ,":", model.feature_importances_)
    
    # #데이터 출력
    submission_csv['count'] = submit
    import time as tm
    ltm = tm.localtime()
    file_name = f'sampleSubmission_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}.csv'
    submission_csv.to_csv(path + file_name, index = False )


'''
[DecisionTreeRegressor] model r2 :  -0.2511529043168965
[DecisionTreeRegressor] mode eval_r2 :  -0.2511529043168965
[DecisionTreeRegressor] mode eval_mse :  40441.17007722665
DecisionTreeRegressor : [0.07038231 0.00861002 0.04439198 0.04575768 0.14182606 0.22691505
 0.26099699 0.20111991]
[RandomForestRegressor] model r2 :  0.2669729583434547
[RandomForestRegressor] mode eval_r2 :  0.2669729583434547
[RandomForestRegressor] mode eval_mse :  23693.723733170656
RandomForestRegressor : [0.06696492 0.00706911 0.04100002 0.05288532 0.14221594 0.23247831
 0.26435576 0.19303062]
[GradientBoostingRegressor] model r2 :  0.3282317877700205
[GradientBoostingRegressor] mode eval_r2 :  0.3282317877700205
[GradientBoostingRegressor] mode eval_mse :  21713.647012712452
GradientBoostingRegressor : [0.07265929 0.0011075  0.03215725 0.01314505 0.20912877 0.28687031
 0.35936456 0.02556727]
[XGBRegressor] model r2 :  0.32819435987379497
[XGBRegressor] mode eval_r2 :  0.32819435987379497
[XGBRegressor] mode eval_mse :  21714.856799231482
XGBRegressor : [0.12411347 0.06257048 0.09058706 0.08034427 0.11615066 0.31688994
 0.14881815 0.06052593]
'''