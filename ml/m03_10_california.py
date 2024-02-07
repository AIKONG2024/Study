from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, random_state=2874458)

# 모델 구성
models = [LinearSVR(),LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(),  RandomForestRegressor()]
for model in models:

    # 컴파일, 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    r2 = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    r2_pred = r2_score(y_test, y_predict)
    result = model.predict(x)
    print(f"[{type(model).__name__}] model r2 : ", r2)
    print(f"[{type(model).__name__}] mode eval_r2 : ", r2_pred)

'''
[LinearSVR] model r2 :  -6.72620981937661
[LinearSVR] mode eval_r2 :  -6.72620981937661
[LinearRegression] model r2 :  0.6133856063974148
[LinearRegression] mode eval_r2 :  0.6133856063974148
[KNeighborsRegressor] model r2 :  0.15627721317983434
[KNeighborsRegressor] mode eval_r2 :  0.15627721317983434
[DecisionTreeRegressor] model r2 :  0.5943225677016997
[DecisionTreeRegressor] mode eval_r2 :  0.5943225677016997
[RandomForestRegressor] model r2 :  0.8106541738490985
[RandomForestRegressor] mode eval_r2 :  0.8106541738490985
'''

