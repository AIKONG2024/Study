from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from xgboost import XGBRegressor
import time
seed = 42
# 1. 데이터
datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.72, random_state=seed) 

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

parameters = {
    'n_estimators': [100, 200, 300, 400],  # 디폴트 100
    'learning_rate': [0.5, 1, 0.01, 0.001],  # 디폴트 0.3 / 0~1 / eta *
    'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 디폴트 0 / 0~inf
    # 'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],  # 디폴트 1 / 0~inf
    # 'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
    # 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
    # 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
    # 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
    # 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],  # 디폴트 0 / 0~inf/ L1 절대값 가중치 규제 / alpha
    # 'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10],  # 디폴트 1/ 0~inf/ L2 제곱 가중치 규제 / lambda
}

# 2. 모델
xgb = XGBRegressor(random_state = seed)
# Hyperparameter Optimization
model = GridSearchCV(xgb, param_grid=parameters , cv=kf, n_jobs=22)
model.fit(X_train, y_train)
x_predictsion = model.best_estimator_.predict(X_test)

results = model.score(X_test, y_test)
best_acc_score = r2_score(y_test, x_predictsion) 
print(
f"""
============================================
[best_acc_score : {best_acc_score}]
[Best params : {model.best_params_}]
[Best accuracy_score: {model.best_score_}]
============================================
"""
)
'''
============================================
[best_acc_score : 0.8366353670848423]
[Best params : {'learning_rate': 0.5, 'max_depth': 3, 'n_estimators': 400}]
[Best accuracy_score: 0.8282044407602307]
============================================
'''
