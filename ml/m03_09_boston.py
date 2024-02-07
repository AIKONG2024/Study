import numpy as np
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

datasets= load_boston()
x = datasets.data
y = datasets.target

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=20)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구성
models = [LinearSVR(), LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(),  RandomForestRegressor()]
for model in models:
    model.fit(x_train, y_train)

    #평가 예측
    from sklearn.metrics import r2_score
    r2 = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    r2_pred = r2_score(y_test, y_predict)
    print(f"[{type(model).__name__}] model r2 : ", r2)
    print(f"[{type(model).__name__}] mode eval_r2 : ", r2_pred)
'''
[LinearSVR] model r2 :  0.5829271219766516
[LinearSVR] mode eval_r2 :  0.5829271219766516
[LinearRegression] model r2 :  0.7023964981707983
[LinearRegression] mode eval_r2 :  0.7023964981707983
[KNeighborsRegressor] model r2 :  0.6943369094988743
[KNeighborsRegressor] mode eval_r2 :  0.6943369094988743
[DecisionTreeRegressor] model r2 :  0.5250107174585736
[DecisionTreeRegressor] mode eval_r2 :  0.5250107174585736
[RandomForestRegressor] model r2 :  0.6913310149656464
[RandomForestRegressor] mode eval_r2 :  0.6913310149656464
'''
