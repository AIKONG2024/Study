import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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

#모델 구성
models = [LinearSVR(), Perceptron(), LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(),  RandomForestRegressor()]
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
    
    # #데이터 출력
    submission_csv['count'] = submit
    import time as tm
    ltm = tm.localtime()
    file_name = f'sampleSubmission_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}.csv'
    submission_csv.to_csv(path + file_name, index = False )


'''
[LinearSVR] model r2 :  0.2201948649918578
[LinearSVR] mode eval_r2 :  0.2201948649918578
[LinearSVR] mode eval_mse :  25205.737830402984
[Perceptron] model r2 :  0.00612369871402327
[Perceptron] mode eval_r2 :  -0.41372898571213557
[Perceptron] mode eval_mse :  45696.13686466626
[LinearRegression] model r2 :  0.2574230473244602
[LinearRegression] mode eval_r2 :  0.2574230473244602
[LinearRegression] mode eval_mse :  24002.406688234732
[KNeighborsRegressor] model r2 :  0.3063187160588716
[KNeighborsRegressor] mode eval_r2 :  0.3063187160588716
[KNeighborsRegressor] mode eval_mse :  22421.945939987752
[DecisionTreeRegressor] model r2 :  -0.24697618121179143
[DecisionTreeRegressor] mode eval_r2 :  -0.24697618121179143
[DecisionTreeRegressor] mode eval_mse :  40306.16533969517
[RandomForestRegressor] model r2 :  0.26565193551130906
[RandomForestRegressor] mode eval_r2 :  0.26565193551130906
[RandomForestRegressor] mode eval_mse :  23736.423317567074
'''