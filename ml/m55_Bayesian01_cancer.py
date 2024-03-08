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
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

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

#2. 모델
bayesian_params = {
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1,50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'min_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, min_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight': int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),
        'colsample_bytree' : colsample_bytree,
        'min_bin' : max(int(round(min_bin)), 10),
        'reg_lambda':max(reg_lambda, 0),
        'reg_alpha' : reg_alpha,
    }
    
    model = XGBClassifier(**params, n_jobs = -1)
    
    model.fit(x_train, y_train,
              eval_set = [(x_train, y_train), (x_test,y_test)],
              eval_metric = 'logloss',
              verbose=0,
              early_stopping_rounds = 50,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds= bayesian_params,
    random_state=42,
)

n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print("bayesian tunning time : ", round(end_time - start_time, 2))

'''
{'target': 1.0, 'params': {'colsample_bytree': 0.5, 'learning_rate': 1.0, 'max_depth': 10.0, 'min_bin': 391.022086279353, 'min_child_samples': 10.0, 'min_child_weight': 1.0, 'num_leaves': 40.0, 'reg_alpha': 0.01, 'reg_lambda': -0.001, 'subsample': 0.5}}
bayesian tunning time :  35.57
'''