from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

# 1. 데이터
path = "C:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

print(train_csv.shape) #(1459, 11)
print(test_csv.shape) #(715, 10)

# 보간법 - 결측치 처리
from sklearn.impute import KNNImputer
#KNN
imputer = KNNImputer(weights='distance')
train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)

# 평가 데이터 분할
x = train_csv.drop(["count"], axis=1).drop(["casual"], axis=1).drop("registered", axis=1)
y = train_csv["count"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings('ignore') #워닝 무시
parameters = {
    'n_estimators': 1000,  # 디폴트 100
    'learning_rate': 0.01,  # 디폴트 0.3 / 0~1 / eta *
    'max_depth': 3,  # 디폴트 0 / 0~inf
    'gamma': 0,
    'min_child_weight' : 0,
    'subsample' : 0.4,
    'colsample_bytree' :0.8,
    'colsample_bylevel' : 0.7,
    'colsample_bynode': 1,
    'reg_alpha': 0,
    'reg_lambda' : 1,
    'random_state' : 42,
}

# 2. 모델 구성
xgb = XGBRegressor()
xgb.set_params(**parameters, eval_metric = 'rmse')
rf = RandomForestRegressor()
lr = LinearRegression()

model = VotingRegressor(
    estimators=[('LR',lr), ('RF',rf), ('XGB', xgb)],
)

print(f'===========voting============')
# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :", result)
x_predict = model.predict(x_test)
acc = r2_score(y_test, x_predict)
print("acc_score :", acc)

print("=================기존==================")
model_class = [xgb, rf, lr]

for model in model_class:
    class_name = model.__class__.__name__
    # 3. 훈련
    model.fit(x_train, y_train)

    # 4. 평가, 예측
    result = model.score(x_test, y_test)
    x_predict = model.predict(x_test)
    score = r2_score(y_test, x_predict)
    print("{0} 점수 : {1:.4f}".format(class_name, score))

'''
===========voting============
최종점수 : 0.33581349294843077
acc_score : 0.33581349294843077
=================기존==================       
XGBRegressor 점수 : 0.3119
RandomForestRegressor 점수 : 0.2533
LinearRegression 점수 : 0.2504
'''