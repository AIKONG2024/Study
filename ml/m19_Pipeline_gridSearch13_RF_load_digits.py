from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time

x,y = load_digits(return_X_y=True)

print(x.shape, y.shape) #(1797, 64) (1797,)
print(pd.value_counts(y, sort=False))

#Random으로 1번만 돌리고
#Grid Search, Randomized Search 로 돌려보기
#시간체크

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
RF Only
accuracy _ score :  0.9916666666666667
최적 튠 ACC : 0.9916666666666667
걸린 시간 :  0.15 초

==============================
GridSearchCV
최적의 매개변수  RandomForestClassifier(n_jobs=-1)
최적의 파라미터  {'min_samples_split': 2, 'n_jobs': -1}
best score :  0.9756436314363144
accuracy _ score :  0.9916666666666667
최적 튠 ACC : 0.9916666666666667
걸린 시간 :  18.58 초

==============================
RandomizedSearchCV
최적의 매개변수  RandomForestClassifier(min_samples_leaf=3, n_jobs=-1)
최적의 파라미터  {'n_jobs': -1, 'min_samples_split': 2, 'min_samples_leaf': 3}
best score :  0.9652076074332172
accuracy _ score :  0.9916666666666667
최적 튠 ACC : 0.9916666666666667
걸린 시간 :  5.19 초
========================
halving

Name: count, dtype: int64
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 48
n_resources: 100
Fitting 5 folds for each of 48 candidates, totalling 240 fits
----------
iter: 1
n_candidates: 16
n_resources: 300
Fitting 5 folds for each of 16 candidates, totalling 80 fits
----------
iter: 2
n_candidates: 6
n_resources: 900
Fitting 5 folds for each of 6 candidates, totalling 30 fits
최적의 매개변수  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
최적의 파라미터  {'min_samples_split': 3, 'n_jobs': -1}
best score :  0.9654500310366233
accuracy _ score :  0.9888888888888889
최적 튠 ACC : 0.9888888888888889
걸린 시간 :  26.42 초
===========
pipeline
score : 0.9888888888888889
걸린 시간 : 0.16 초
=============
Pipeline + gridsearch
Fitting 5 folds for each of 28 candidates, totalling 140 fits
걸린 시간 : 3.94 초
best_model_acc_score :  0.9916666666666667

최적의 파라미터 :       Pipeline(steps=[('MM', MinMaxScaler()),
                ('RF', RandomForestClassifier(n_jobs=-1))])
최적의 매개변수 :       {'RF__min_samples_split': 2, 'RF__n_jobs': -1}
best score :            0.9672933604336043
best_model_acc_score :  0.9916666666666667
'''