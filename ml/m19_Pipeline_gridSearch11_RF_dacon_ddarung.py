# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, HalvingGridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null

x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
y = train_csv['count'] 

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
최적의 파라미터 :       RandomForestRegressor(min_samples_split=5)
최적의 매개변수 :       {'min_samples_split': 5}
best score :            0.7604715455180704
best_model_acc_score :  0.7876307388287278

Fitting 5 folds for each of 60 candidates, totalling 300 fits
걸린 시간 : 6.44 초
best_model_acc_score :  0.8045118307978225

최적의 파라미터 :       RandomForestRegressor(n_jobs=4)
최적의 매개변수 :       {'min_samples_split': 2, 'n_jobs': 4}
best score :            0.7610412772577533
best_model_acc_score :  0.8045118307978225
===================
randSerchcv
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 3.69 초
best_model_acc_score :  0.7953026081806237

최적의 파라미터 :       RandomForestRegressor(min_samples_split=3)
최적의 매개변수 :       {'min_samples_split': 3}
best score :            0.7596904569478808
best_model_acc_score :  0.7953026081806237
======================
halving

n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 43
max_resources_: 1167
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 60
n_resources: 43
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 20
n_resources: 129
Fitting 5 folds for each of 20 candidates, totalling 100 fits
----------
iter: 2
n_candidates: 7
n_resources: 387
Fitting 5 folds for each of 7 candidates, totalling 35 fits
----------
iter: 3
n_candidates: 3
n_resources: 1161
Fitting 5 folds for each of 3 candidates, totalling 15 fits
걸린 시간 : 5.87 초
best_model_acc_score :  0.7921970045303547

최적의 파라미터 :       RandomForestRegressor()
최적의 매개변수 :       {'min_samples_split': 2}
best score :            0.7593900353209256
best_model_acc_score :  0.7921970045303547
===============
pipeline
score : 0.7875896988246799
걸린 시간 : 0.26 초
=============
Pipeline + gridsearch
Fitting 5 folds for each of 28 candidates, totalling 140 fits
걸린 시간 : 4.8 초
best_model_acc_score :  0.7979665759083872

최적의 파라미터 :       Pipeline(steps=[('MM', MinMaxScaler()),
                ('RF', RandomForestRegressor(min_samples_split=3, n_jobs=-1))])
최적의 매개변수 :       {'RF__min_samples_split': 3, 'RF__n_jobs': -1}
best score :            0.7611076874739233
best_model_acc_score :  0.7979665759083872
'''