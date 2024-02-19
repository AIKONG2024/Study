from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
from xgboost import XGBClassifier

seed = 42
datasets = fetch_covtype()
X = datasets.data
y = datasets.target

lbe = LabelEncoder()
y = lbe.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.72, random_state=123, stratify=y)

scaler = MinMaxScaler()
# scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

# 'n_estimators': [100, 200, 300, 400, 500, 1000],  # 디폴트 100
# 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 0.01, 0.001],  # 디폴트 0.3 / 0~1 / eta *
# 'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 디폴트 0 / 0~inf
# 'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],  # 디폴트 1 / 0~inf
# 'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
# 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],  # 디폴트 0 / 0~inf/ L1 절대값 가중치 규제 / alpha
# 'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10],  # 디폴트 1/ 0~inf/ L2 제곱 가중치 규제 / lambda

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
xgb = XGBClassifier(random_state = seed)
# Hyperparameter Optimization
model = GridSearchCV(xgb, param_grid=parameters , cv=kf, n_jobs=22)
model.fit(X_train, y_train)
x_predictsion = lbe.inverse_transform(model.best_estimator_.predict(X_test)) 

results = model.score(X_test, y_test)
best_acc_score = accuracy_score(y_test, x_predictsion) 
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
[best_acc_score : 0.013996459393671166]
[Best params : {'learning_rate': 1, 'max_depth': None, 'n_estimators': 400}]
[Best accuracy_score: 0.9563524310289964]
============================================
'''
