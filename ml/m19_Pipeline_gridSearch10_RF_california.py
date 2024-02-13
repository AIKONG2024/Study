import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score,cross_val_predict
from sklearn.experimental import enable_halving_search_cv
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = fetch_california_housing(return_X_y=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV,RandomizedSearchCV, HalvingGridSearchCV

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
최적의 파라미터 :       RandomForestRegressor(min_samples_split=3, n_jobs=-1)
최적의 매개변수 :       {'min_samples_split': 3, 'n_jobs': -1}
best score :            0.8039112943592761
best_model_acc_score :  0.8134536895590825

Fitting 5 folds for each of 60 candidates, totalling 300 fits
걸린 시간 : 72.46 초
best_model_acc_score :  0.8154475136783064

최적의 파라미터 :       RandomForestRegressor(min_samples_split=3)
최적의 매개변수 :       {'min_samples_split': 3}
best score :            0.8038639575439561
best_model_acc_score :  0.8154475136783064
=========================
randsearch
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 18.93 초
best_model_acc_score :  0.813738022428296

최적의 파라미터 :       RandomForestRegressor(min_samples_split=3)
최적의 매개변수 :       {'min_samples_split': 3}
best score :            0.8026604995320682
best_model_acc_score :  0.813738022428296
========================
halving

n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 611
max_resources_: 16512
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 60
n_resources: 611
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 20
n_resources: 1833
Fitting 5 folds for each of 20 candidates, totalling 100 fits
----------
iter: 2
n_candidates: 7
n_resources: 5499
Fitting 5 folds for each of 7 candidates, totalling 35 fits
----------
iter: 3
n_candidates: 3
n_resources: 16497
Fitting 5 folds for each of 3 candidates, totalling 15 fits
걸린 시간 : 18.39 초
best_model_acc_score :  0.8120162316086589

최적의 파라미터 :       RandomForestRegressor(n_jobs=4)
최적의 매개변수 :       {'min_samples_split': 2, 'n_jobs': 4}
best score :            0.804038788884478
best_model_acc_score :  0.8120162316086589
==========
pipeline
score : 0.8134346817879048
걸린 시간 : 5.7 초
=============
Pipeline + gridsearch
Fitting 5 folds for each of 28 candidates, totalling 140 fits
걸린 시간 : 35.26 초
best_model_acc_score :  0.8134429105984298

최적의 파라미터 :       Pipeline(steps=[('MM', MinMaxScaler()),
                ('RF', RandomForestRegressor(min_samples_split=5, n_jobs=-1))])
최적의 매개변수 :       {'RF__min_samples_split': 5, 'RF__n_jobs': -1}
best score :            0.8024890901890203
best_model_acc_score :  0.8134429105984298
'''