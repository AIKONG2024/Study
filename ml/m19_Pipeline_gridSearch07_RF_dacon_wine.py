# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,cross_val_predict, HalvingGridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV,RandomizedSearchCV

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
parameters = [
    {"RF__n_jobs" : [-1], "RF__n_estimators": [100, 200],  "RF__min_samples_leaf": [3, 10]},
    {"RF__n_jobs" : [-1], "RF__min_samples_leaf": [3, 5, 7, 10]},
    {"RF__n_jobs" : [-1], "RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]},
    {"RF__n_jobs" : [-1], "RF__min_samples_split": [2, 3, 5, 10]},
]

from sklearn.pipeline import Pipeline
pipe = Pipeline([("MM", MinMaxScaler()), 
                  ("RF", RandomForestClassifier())])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1,  n_jobs= -1 )
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1,  n_jobs= -1 )
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1,  n_jobs= -1 )

import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print("걸린 시간 :", round(end_time - start_time ,2 ), "초")


from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("best_model_acc_score : ", best_acc_score) #best_acc_score :  0.9333333333333333

print(f'''
최적의 파라미터 :\t{model.best_estimator_}
최적의 매개변수 :\t{model.best_params_}
best score :\t\t{model.best_score_}
best_model_acc_score :\t{best_acc_score}
''')



'''
최적의 파라미터 :       RandomForestClassifier(min_samples_split=3, n_jobs=-1)
최적의 매개변수 :       {'min_samples_split': 3, 'n_jobs': -1}
best score :            0.6502174475126694
best_model_acc_score :  0.6790909090909091

Fitting 5 folds for each of 60 candidates, totalling 300 fits
걸린 시간 : 9.4 초
best_model_acc_score :  0.6627272727272727

최적의 파라미터 :       RandomForestClassifier(min_samples_split=5, n_jobs=4)
최적의 매개변수 :       {'min_samples_split': 5, 'n_jobs': 4}
best score :            0.6536311924707829
best_model_acc_score :  0.6627272727272727
======================
rdsearch
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 4.1 초
best_model_acc_score :  0.6745454545454546

최적의 파라미터 :       RandomForestClassifier(n_jobs=-1)
최적의 매개변수 :       {'n_jobs': -1, 'min_samples_split': 2}
best score :            0.6490805667597477
best_model_acc_score :  0.6745454545454546
=======================
halving

n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 162
max_resources_: 4397
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 60
n_resources: 162
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 20
n_resources: 486
Fitting 5 folds for each of 20 candidates, totalling 100 fits
----------
iter: 2
n_candidates: 7
n_resources: 1458
Fitting 5 folds for each of 7 candidates, totalling 35 fits
----------
iter: 3
n_candidates: 3
n_resources: 4374
Fitting 5 folds for each of 3 candidates, totalling 15 fits
걸린 시간 : 6.56 초
best_model_acc_score :  0.6318181818181818

최적의 파라미터 :       RandomForestClassifier(max_depth=12, min_samples_leaf=7)
최적의 매개변수 :       {'max_depth': 12, 'min_samples_leaf': 7}
best score :            0.6015584177835894
best_model_acc_score :  0.6318181818181818
===============
pipeline
score : 0.6863636363636364
걸린 시간 : 0.46 초
=================
Pipeline+gridsearch
걸린 시간 : 6.55 초
best_model_acc_score :  0.68

최적의 파라미터 :       Pipeline(steps=[('MM', MinMaxScaler()),
                ('RF', RandomForestClassifier(n_jobs=-1))])
최적의 매개변수 :       {'RF__min_samples_split': 2, 'RF__n_jobs': -1}
best score :            0.6604550625711035
best_model_acc_score :  0.68
'''