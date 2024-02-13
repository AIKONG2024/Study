#https://dacon.io/competitions/open/236070/
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,cross_val_predict, HalvingGridSearchCV
from sklearn.utils import all_estimators
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

path = "C:/_data/dacon/iris/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

#데이터 확인
x = train_csv.drop(columns='species')
y = train_csv['species']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV ,RandomizedSearchCV

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

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
# model = HalvingGridSearchCV(rfc, parameters, cv=kf , n_jobs=-1, refit=True, verbose=1,random_state=42,factor=3, min_resources=10)
from sklearn.pipeline import make_pipeline
model = make_pipeline(MinMaxScaler(), RandomForestClassifier(min_samples_split= 2))

import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print("걸린 시간 :", round(end_time - start_time ,2 ), "초")

from sklearn.metrics import accuracy_score
# best_predict = model.best_estimator_.predict(x_test)
# best_acc_score = accuracy_score(y_test, best_predict)

# print("best_model_acc_score : ", best_acc_score) #best_acc_score :  0.9333333333333333
predict = model.predict(x_test)

print(f'''
score : {accuracy_score(y_test, predict)}
걸린 시간 : {round(end_time - start_time ,2 )} 초
''')



# print(f'''
# 최적의 파라미터 :\t{model.best_estimator_}
# 최적의 매개변수 :\t{model.best_params_}
# best score :\t\t{model.best_score_}
# best_model_acc_score :\t{best_acc_score}
# ''')



'''
최적의 파라미터 :       RandomForestClassifier(max_depth=12, min_samples_leaf=3, n_estimators=200)
최적의 매개변수 :       {'max_depth': 12, 'min_samples_leaf': 3, 'n_estimators': 200}
best score :            0.9694736842105263
best_model_acc_score :  0.9166666666666666

Fitting 5 folds for each of 60 candidates, totalling 300 fits
걸린 시간 : 4.32 초
best_model_acc_score :  0.9166666666666666

최적의 파라미터 :       RandomForestClassifier(max_depth=6, min_samples_leaf=10, n_estimators=200)
최적의 매개변수 :       {'max_depth': 6, 'min_samples_leaf': 10, 'n_estimators': 200}
best score :            0.9694736842105263
best_model_acc_score :  0.9166666666666666
======================
rdSearch
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 3.15 초
best_model_acc_score :  0.9166666666666666

최적의 파라미터 :       RandomForestClassifier(min_samples_leaf=7)
최적의 매개변수 :       {'min_samples_split': 2, 'min_samples_leaf': 7}
best score :            0.9594736842105263
best_model_acc_score :  0.9166666666666666
======================
halving

n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 10
max_resources_: 96
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
걸린 시간 : 5.29 초
best_model_acc_score :  0.9166666666666666

최적의 파라미터 :       RandomForestClassifier(min_samples_split=3, n_jobs=4)
최적의 매개변수 :       {'min_samples_split': 3, 'n_jobs': 4}
best score :            0.9660130718954247
best_model_acc_score :  0.9166666666666666
===================
pipeline
score : 0.9166666666666666
걸린 시간 : 0.05 초
'''