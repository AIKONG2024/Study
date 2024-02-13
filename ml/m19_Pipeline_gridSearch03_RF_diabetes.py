from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
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
parameters = [
    {"RF__n_jobs" : [-1], "RF__n_estimators": [100, 200],  "RF__min_samples_leaf": [3, 10]},
    {"RF__n_jobs" : [-1], "RF__min_samples_leaf": [3, 5, 7, 10]},
    {"RF__n_jobs" : [-1], "RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]},
    {"RF__n_jobs" : [-1], "RF__min_samples_split": [2, 3, 5, 10]},
]

from sklearn.pipeline import Pipeline
pipe = Pipeline([("MM", MinMaxScaler()), 
                  ("RF", RandomForestRegressor())])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1,  n_jobs= -1 )
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1,  n_jobs= -1 )
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1,  n_jobs= -1 )

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

========================
halving
n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 13
max_resources_: 353
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 60
n_resources: 13
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 20
n_resources: 39
Fitting 5 folds for each of 20 candidates, totalling 100 fits
----------
iter: 2
n_candidates: 7
n_resources: 117
Fitting 5 folds for each of 7 candidates, totalling 35 fits
----------
iter: 3
n_candidates: 3
n_resources: 351
Fitting 5 folds for each of 3 candidates, totalling 15 fits
걸린 시간 : 4.56 초
best_model_acc_score :  0.5513476135795226
=================
pipeline
score : 0.5578275223037032
걸린 시간 : 0.15 초
==================
Pipeline + gridsearch
걸린 시간 : 3.28 초
best_model_acc_score :  0.550671678522747

최적의 파라미터 :       Pipeline(steps=[('MM', MinMaxScaler()),
                ('RF',
                 RandomForestRegressor(min_samples_leaf=5, min_samples_split=3,
                                       n_jobs=-1))])
최적의 매개변수 :       {'RF__min_samples_leaf': 5, 'RF__min_samples_split': 3, 'RF__n_jobs': -1}
best score :            0.4296019001254628
best_model_acc_score :  0.550671678522747

'''