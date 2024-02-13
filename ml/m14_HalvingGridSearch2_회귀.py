import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV
)
from sklearn.datasets import load_iris,load_digits, load_diabetes
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)
import time


import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
# x,y = load_iris(return_X_y=True)
# print(x.shape) #(150, 4)
# print(y.shape)#(150,)
# x,y = load_digits(return_X_y=True)
# print(x.shape, y.shape)#(1797, 64) (1797,)
x,y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)#(1797, 64) (1797,)

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
)
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"C": [1, 10, 100, 1000], "kernel": ["linear"], "degree": [3, 4, 5]}, #12
    {"C": [1, 10, 100], "kernel": ["rbf"], "gamma": [0.001, 0.0001]}, #6
    {"C": [1, 10, 100, 1000],"kernel": ["sigmoid"],"gamma": [0.01, 0.001, 0.0001],"degree": [3, 4],  #24
    },
]

# 2. 모델
# model = SVC(C=1, kernel="linear", degree=3)
# model = GridSearchCV(SVC(), param_grid=parameters, cv=kf, verbose=1, refit=True, n_jobs= -1 )
# model = RandomizedSearchCV(SVC(), parameters, cv=kf, verbose=1, refit=True, random_state=66, n_iter=20)
# model = HalvingGridSearchCV(SVC(), parameters, cv=3, verbose=1, refit=True, random_state=66, factor=4, min_resources=150) 
model = HalvingGridSearchCV(SVR(), parameters, cv=3, verbose=1, refit=True, random_state=66, factor=3)
#수정 ) 회귀에서 resaurce의 개수 : cv*2 + alpha 

'''
==========================iris
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 42
n_resources: 30
Fitting 5 folds for each of 42 candidates, totalling 210 fits
----------
iter: 1
n_candidates: 14
n_resources: 90
Fitting 5 folds for each of 14 candidates, totalling 70 fits
'''


start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 ", model.best_estimator_) #가장 좋은 parameter 빼줌.
# 최적의 매개변수  SVC(C=1, kernel='linear')
print("최적의 파라미터 ", model.best_params_)#지정한 parameter 에서 가장 좋은 것 빼줌.
# 최적의 파라미터  {'C': 1, 'degree': 3, 'kernel': 'linear'}
print("best score : ", model.best_score_) #train 의 Score
# best score :  0.975
print("model score : ", model.score(x_test, y_test))
# model score :  0.9666666666666667

y_predict = model.predict(x_test)
acc_score = r2_score(y_test, y_predict)
print("r2 _ score : ", acc_score)
#accuracy _ score :  0.9666666666666667

y_predict_best = model.best_estimator_.predict(x_test)
print("최적 튠 R2 :", r2_score(y_test, y_predict_best))

print("걸린 시간 : ", round(end_time - start_time ,2 ),"초") #걸린 시간 :  2.14 초

