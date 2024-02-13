import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

x,y = load_wine(return_X_y=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

from sklearn.pipeline import make_pipeline
model = make_pipeline(MinMaxScaler(), RandomForestClassifier(max_depth=10, min_samples_leaf=3, n_estimators=200))

# n_splits = 5
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# parameters = [
#     {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
#     {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
#     {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
#     {"min_samples_split": [2, 3, 5, 10]},
#     {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
# ]
# rfc = RandomForestClassifier()
# model = HalvingGridSearchCV(rfc, parameters, cv=kf , n_jobs=-1, refit=True, verbose=1 ,random_state=42, factor=3, min_resources=10)
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
# print("걸린 시간 :", round(end_time - start_time ,2 ), "초")
predict = model.predict(x_test)

print(f'''
score : {accuracy_score(y_test, predict)}
걸린 시간 : {round(end_time - start_time ,2 )} 초
''')


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

'''
최적의 파라미터 :       RandomForestClassifier(max_depth=10, min_samples_leaf=3, n_estimators=200)
최적의 매개변수 :       {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 200}
best score :            0.993103448275862
best_model_acc_score :  0.9722222222222222

Fitting 5 folds for each of 60 candidates, totalling 300 fits
걸린 시간 : 3.53 초
best_model_acc_score :  0.9722222222222222

최적의 파라미터 :       RandomForestClassifier(max_depth=8, min_samples_leaf=3)
최적의 매개변수 :       {'max_depth': 8, 'min_samples_leaf': 3}
best score :            0.993103448275862
best_model_acc_score :  0.9722222222222222
====================
rdsearch 

Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 2.42 초
best_model_acc_score :  0.9722222222222222

최적의 파라미터 :       RandomForestClassifier(max_depth=10, min_samples_leaf=3, n_estimators=200)
최적의 매개변수 :       {'n_estimators': 200, 'min_samples_leaf': 3, 'max_depth': 10}
best score :            0.9862068965517242
best_model_acc_score :  0.9722222222222222
========================
halving

n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 10
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 60
n_resources: 10
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 20
n_resources: 30
Fitting 5 folds for each of 20 candidates, totalling 100 fits
----------
iter: 2
n_candidates: 7
n_resources: 90
Fitting 5 folds for each of 7 candidates, totalling 35 fits
걸린 시간 : 4.36 초
best_model_acc_score :  0.9722222222222222

최적의 파라미터 :       RandomForestClassifier(min_samples_leaf=5, min_samples_split=3)
최적의 매개변수 :       {'min_samples_leaf': 5, 'min_samples_split': 3}
best score :            0.9888888888888889
best_model_acc_score :  0.9722222222222222
=================
pipeline
score : 0.9722222222222222
걸린 시간 : 0.1 초
'''