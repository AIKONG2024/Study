from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
]
rfc = RandomForestRegressor()
model = RandomizedSearchCV(rfc, parameters, cv=kf , n_jobs=-1, refit=True, verbose=1)
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print("걸린 시간 :", round(end_time - start_time ,2 ), "초")

from sklearn.metrics import r2_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = r2_score(y_test, best_predict)

print("best_model_acc_score : ", best_acc_score) #best_acc_score :  0.9333333333333333

print(f'''
최적의 파라미터 :\t{model.best_estimator_}
최적의 매개변수 :\t{model.best_params_}
best score :\t\t{model.best_score_}
best_model_acc_score :\t{best_acc_score}
''')

'''
최적의 파라미터 :       RandomForestRegressor(min_samples_leaf=10)
최적의 매개변수 :       {'min_samples_leaf': 10, 'min_samples_split': 2}
best score :            0.41358555127487423
best_model_acc_score :  0.5665789288865843

Fitting 5 folds for each of 60 candidates, totalling 300 fits
걸린 시간 : 3.87 초
best_model_acc_score :  0.550330929653679

최적의 파라미터 :       RandomForestRegressor(max_depth=8, min_samples_leaf=10)
최적의 매개변수 :       {'max_depth': 8, 'min_samples_leaf': 10}
best score :            0.41378953154207077
best_model_acc_score :  0.550330929653679

======================
rdsearch
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 2.54 초
best_model_acc_score :  0.557274657387024

최적의 파라미터 :       RandomForestRegressor(max_depth=12, min_samples_leaf=10, n_estimators=200)
최적의 매개변수 :       {'n_estimators': 200, 'min_samples_leaf': 10, 'max_depth': 12}
best score :            0.41121597013663286
best_model_acc_score :  0.557274657387024

'''