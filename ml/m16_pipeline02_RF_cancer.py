import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold ,cross_val_score, StratifiedKFold, cross_val_predict, HalvingGridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import time
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_breast_cancer(return_X_y=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# n_splits = 5
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# parameters = [
#     {"n_jobs" : [-1], "n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
#     {"n_jobs" : [-1], "max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
#     {"n_jobs" : [-1], "min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
#     {"n_jobs" : [-1], "min_samples_split": [2, 3, 5, 10]},
# ]
# rfc = RandomForestClassifier()
# model = HalvingGridSearchCV(rfc, parameters, cv=kf , n_jobs=-1, refit=True, verbose=1, factor=3, min_resources=15)
from sklearn.pipeline import make_pipeline
model = make_pipeline(MinMaxScaler(), RandomForestClassifier(min_samples_split=3, n_jobs=-1))
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
# print("걸린 시간 :", round(end_time - start_time ,2 ), "초")


# from sklearn.metrics import accuracy_score
# best_predict = model.best_estimator_.predict(x_test)
# best_acc_score = accuracy_score(y_test, best_predict)

# print("best_model_acc_score : ", best_acc_score) #best_acc_score :  0.9333333333333333

# print(f'''
# 최적의 파라미터 :\t{model.best_estimator_}
# 최적의 매개변수 :\t{model.best_params_}
# best score :\t\t{model.best_score_}
# best_model_acc_score :\t{best_acc_score}
# ''')
predict = model.predict(x_test)

print(f'''
score : {accuracy_score(y_test, predict)}
걸린 시간 : {round(end_time - start_time ,2 )} 초
''')



'''
Fitting 5 folds for each of 48 candidates, totalling 240 fits
최적의 파라미터 :       RandomForestClassifier(max_depth=12, min_samples_leaf=3, n_estimators=200)
최적의 매개변수 :       {'max_depth': 12, 'min_samples_leaf': 3, 'n_estimators': 200}
best score :            0.9670329670329672
best_model_acc_score :  0.9649122807017544
걸린 시간 : 4.25 초

===============================
RandomizeSearch
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 2.62 초
best_model_acc_score :  0.956140350877193

최적의 파라미터 :       RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_jobs=-1)
최적의 매개변수 :       {'n_jobs': -1, 'min_samples_leaf': 3, 'max_depth': 6}
best score :            0.9626373626373628
best_model_acc_score :  0.956140350877193
===========================
halving

n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 15
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 48
n_resources: 15
Fitting 5 folds for each of 48 candidates, totalling 240 fits
----------
iter: 1
n_candidates: 16
n_resources: 45
Fitting 5 folds for each of 16 candidates, totalling 80 fits
----------
iter: 2
n_candidates: 6
n_resources: 135
Fitting 5 folds for each of 6 candidates, totalling 30 fits
----------
iter: 3
n_candidates: 2
n_resources: 405
Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간 : 5.08 초
best_model_acc_score :  0.9736842105263158

최적의 파라미터 :       RandomForestClassifier(min_samples_split=3, n_jobs=-1)
최적의 매개변수 :       {'min_samples_split': 3, 'n_jobs': -1}
best score :            0.962962962962963
best_model_acc_score :  0.9736842105263158
=================
pipeline
score : 0.956140350877193
걸린 시간 : 0.07 초
'''