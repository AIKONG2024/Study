import pandas as pd
from sklearn.utils import all_estimators
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, HalvingGridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = 'C:/_data/kaggle/bike/'
train_csv =pd.read_csv(path + 'train.csv', index_col=0)
test_csv =pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

#데이터 전처리
x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
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
최적의 파라미터 :       RandomForestRegressor(max_depth=10, min_samples_leaf=3)
최적의 매개변수 :       {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 100}
best score :            0.3540999319432574
best_model_acc_score :  0.3495764269208269

Fitting 5 folds for each of 60 candidates, totalling 300 fits
걸린 시간 : 15.37 초
best_model_acc_score :  0.35269586540014075

최적의 파라미터 :       RandomForestRegressor(max_depth=10, min_samples_leaf=3, n_estimators=200)
최적의 매개변수 :       {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 200}
best score :            0.3539813398673958
best_model_acc_score :  0.35269586540014075

==========================
randsearch
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 6.06 초
best_model_acc_score :  0.35027449451019343

최적의 파라미터 :       RandomForestRegressor(max_depth=10, min_samples_leaf=3, n_estimators=200)
최적의 매개변수 :       {'n_estimators': 200, 'min_samples_leaf': 3, 'max_depth': 10}
best score :            0.35485835692618206
best_model_acc_score :  0.35027449451019343
========================
halving
n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 322
max_resources_: 8708
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 60
n_resources: 322
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 20
n_resources: 966
Fitting 5 folds for each of 20 candidates, totalling 100 fits
----------
iter: 2
n_candidates: 7
n_resources: 2898
Fitting 5 folds for each of 7 candidates, totalling 35 fits
----------
iter: 3
n_candidates: 3
n_resources: 8694
Fitting 5 folds for each of 3 candidates, totalling 15 fits
걸린 시간 : 8.52 초
best_model_acc_score :  0.3399088940167928

최적의 파라미터 :       RandomForestRegressor(max_depth=12, min_samples_leaf=10, n_estimators=200)
최적의 매개변수 :       {'max_depth': 12, 'min_samples_leaf': 10, 'n_estimators': 200}
best score :            0.3528352320932907
best_model_acc_score :  0.3399088940167928
===============
pipeline
score : 0.271579569248809
걸린 시간 : 1.03 초
=============
Pipeline + gridsearch
Fitting 5 folds for each of 28 candidates, totalling 140 fits
걸린 시간 : 8.69 초
best_model_acc_score :  0.34159357220238995

최적의 파라미터 :       Pipeline(steps=[('MM', MinMaxScaler()),
                ('RF',
                 RandomForestRegressor(min_samples_leaf=7, min_samples_split=3,
                                       n_jobs=-1))])
최적의 매개변수 :       {'RF__min_samples_leaf': 7, 'RF__min_samples_split': 3, 'RF__n_jobs': -1}
best score :            0.3549253447222131
best_model_acc_score :  0.34159357220238995
'''