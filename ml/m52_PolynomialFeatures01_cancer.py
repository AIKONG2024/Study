import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

# 1. 데이터
load_cancer = load_breast_cancer()
x = load_cancer.data
y = load_cancer.target

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias= False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
xgb = XGBClassifier()
xgb.set_params(**parameters, eval_metric = 'logloss')
rf = RandomForestClassifier()
lr = LogisticRegression()

models = [xgb,rf, lr]
for model in models :
    model.fit(x_train, y_train)
    class_name = model.__class__.__name__ 
    score = model.score(x_test,y_test)
    print("{0} ACC : {1:.4f}".format(class_name, score))

'''
XGBClassifier ACC : 0.9825
RandomForestClassifier ACC : 0.9737
LogisticRegression ACC : 0.9737

폴리노미알 피처 적용
XGBClassifier ACC : 0.9912
RandomForestClassifier ACC : 0.9825
LogisticRegression ACC : 0.9912
'''