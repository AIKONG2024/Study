import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression

# 1. 데이터
load_cancer = load_breast_cancer()
x = load_cancer.data
y = load_cancer.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# parameters = {
#     'n_estimators': 1000,  # 디폴트 100
#     'learning_rate': 0.01,  # 디폴트 0.3 / 0~1 / eta *
#     'max_depth': 3,  # 디폴트 0 / 0~inf
#     'gamma': 0,
#     'min_child_weight' : 0,
#     'subsample' : 0.4,
#     'colsample_bytree' :0.8,
#     'colsample_bylevel' : 0.7,
#     'colsample_bynode': 1,
#     'reg_alpha': 0,
#     'reg_lambda' : 1,
#     'random_state' : 3377,
# }
# 2. 모델 구성
xgb = XGBClassifier()
# xgb.set_params(**parameters, eval_metric = 'logloss')
model = BaggingClassifier(
    LogisticRegression(),
    n_estimators=10,
    n_jobs=1,
    random_state=777,
    #   bootstrap=True, #디폴트, True : 중복을 허용
    bootstrap=False, 
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :", result)
x_predict = model.predict(x_test)
acc = accuracy_score(y_test, x_predict)
print("acc_score :", acc)

# 기존 xgb bagging acc_score : 0.9824561403508771
# logistic regression bagging acc_score : 0.9649122807017544
