import numpy as np
from sklearn.datasets import load_boston
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, HalvingGridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

datasets= load_boston()
x = datasets.data
y = datasets.target

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV,RandomizedSearchCV

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
최적의 파라미터 :       RandomForestRegressor(n_jobs=-1)
최적의 매개변수 :       {'min_samples_split': 2, 'n_jobs': -1}
best score :            0.8574265827431973
best_model_acc_score :  0.7903848069966082

Fitting 5 folds for each of 60 candidates, totalling 300 fits
걸린 시간 : 4.29 초
best_model_acc_score :  0.777612886428204

최적의 파라미터 :       RandomForestRegressor(min_samples_split=3)
최적의 매개변수 :       {'min_samples_split': 3}
best score :            0.8564366007785301
best_model_acc_score :  0.777612886428204
====================
randsearcv
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 2.64 초
best_model_acc_score :  0.7826322483486509

최적의 파라미터 :       RandomForestRegressor(min_samples_split=3)
최적의 매개변수 :       {'min_samples_split': 3}
best score :            0.8569929458900368
best_model_acc_score :  0.7826322483486509

=====================
halving

n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 14
max_resources_: 404
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 60
n_resources: 14
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 20
n_resources: 42
Fitting 5 folds for each of 20 candidates, totalling 100 fits
----------
iter: 2
n_candidates: 7
n_resources: 126
Fitting 5 folds for each of 7 candidates, totalling 35 fits
----------
iter: 3
n_candidates: 3
n_resources: 378
Fitting 5 folds for each of 3 candidates, totalling 15 fits
걸린 시간 : 4.39 초
best_model_acc_score :  0.7827653374579508

최적의 파라미터 :       RandomForestRegressor(n_jobs=4)
최적의 매개변수 :       {'min_samples_split': 2, 'n_jobs': 4}
best score :            0.8510731316084771
best_model_acc_score :  0.7827653374579508
===================
pipeline
score : 0.757287180919507
걸린 시간 : 0.13 초
=============
Pipeline + gridsearch
Fitting 5 folds for each of 28 candidates, totalling 140 fits
걸린 시간 : 3.39 초
best_model_acc_score :  0.7737530283109746

최적의 파라미터 :       Pipeline(steps=[('MM', MinMaxScaler()),
                ('RF', RandomForestRegressor(n_jobs=-1))])
최적의 매개변수 :       {'RF__min_samples_split': 2, 'RF__n_jobs': -1}
best score :            0.8577887688137966
best_model_acc_score :  0.7737530283109746
'''