from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import numpy as np
from xgboost import XGBRegressor
seed = 42
print()

# 1. 데이터
datasets = load_diabetes()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.72, random_state=seed) 

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

parameters = {
    'n_estimators': 4000,  # 디폴트 100
    'learning_rate': 0.01,  # 디폴트 0.3 / 0~1 / eta *
    'max_depth': 2,  # 디폴트 0 / 0~inf
    'early_stopping_rounds' : 100
    
}

# 2. 모델
model = XGBRegressor(random_state = seed, **parameters)

model.fit(X_train, y_train, eval_set = [(X_test, y_test)], eval_metric = "logloss") #rmse> mae> logloss > error
#회귀 : rmse, rmsle, mae, mape, mphe , map
#문서 : https://xgboost.readthedocs.io/en/stable/parameter.html

print("파라미터 : ",model.get_params())
results = model.score(X_test, y_test)
x_predictsion = model.predict(X_test)
best_score = r2_score(y_test, x_predictsion) 
print(
f"""
============================================
[Best r2_score: {best_score}]
============================================
"""
)
