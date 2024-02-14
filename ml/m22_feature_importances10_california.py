from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, random_state=2874458)


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
    result = model.predict(x)
    print(f"[{type(model).__name__}] model r2 : ", r2)
    print(f"[{type(model).__name__}] mode eval_r2 : ", r2_pred)
    print(type(model).__name__ ,":", model.feature_importances_)

'''
[DecisionTreeRegressor] model r2 :  0.5989790151643832
[DecisionTreeRegressor] mode eval_r2 :  0.5989790151643832
DecisionTreeRegressor : [0.5133933  0.05227543 0.05203381 0.02743027 0.03249431 0.1297271
 0.10036182 0.09228397]
[RandomForestRegressor] model r2 :  0.8080703149030777
[RandomForestRegressor] mode eval_r2 :  0.8080703149030777
RandomForestRegressor : [0.51735869 0.05419302 0.04899904 0.03022036 0.03247766 0.13585979
 0.09185605 0.0890354 ]
[GradientBoostingRegressor] model r2 :  0.7894840719005065
[GradientBoostingRegressor] mode eval_r2 :  0.7894840719005065
GradientBoostingRegressor : [0.59220049 0.03171004 0.02372552 0.00534542 0.0040449  0.12981622
 0.08922621 0.12393119]
[XGBRegressor] model r2 :  0.8320529600599189
[XGBRegressor] mode eval_r2 :  0.8320529600599189
XGBRegressor : [0.49460176 0.06525674 0.04517365 0.02432704 0.02271404 0.14363027
 0.0943241  0.10997245]
'''

