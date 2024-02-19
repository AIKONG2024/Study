from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    train_test_split,
    KFold, StratifiedKFold,
    GridSearchCV,RandomizedSearchCV,
    HalvingGridSearchCV, HalvingRandomSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
seed = 42
#1. 데이터
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

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
    'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],  # 디폴트 1 / 0~inf
    'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
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
x_predictsion = model.best_estimator_.predict(X_test)

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
[best_acc_score : 0.9590643274853801]
[Best params : {'learning_rate': 1, 'max_depth': 3, 'min_child_weight': 0.001, 'n_estimators': 100, 'subsample': 0.2}]
[Best accuracy_score: 0.9799367088607595]
============================================
'''