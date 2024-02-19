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

parameters = {
    'n_estimators': 4000,  # 디폴트 100
    'learning_rate': 0.01,  # 디폴트 0.3 / 0~1 / eta *
    'max_depth': 2,  # 디폴트 0 / 0~inf
    'early_stopping_rounds' : 100
}

# 2. 모델
model = XGBClassifier(random_state = seed, **parameters)
model.fit(X_train, y_train, eval_set = [(X_test, y_test)], eval_metric = "logloss") #
#분류 : logloss , error, auc
#문서 : https://xgboost.readthedocs.io/en/stable/parameter.html

print("파라미터 : ",model.get_params())
results = model.score(X_test, y_test)
x_predictsion = model.predict(X_test)
best_score = accuracy_score(y_test, x_predictsion) 
print(
f"""
============================================
[Best acc_score: {best_score}]
============================================
"""
)
