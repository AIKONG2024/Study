import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV
)
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)
import time


import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
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
model = GridSearchCV(SVC(), param_grid=parameters, cv=kf, verbose=1, refit=True, n_jobs= -1 )
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
acc_score = accuracy_score(y_test, y_predict)
print("accuracy _ score : ", acc_score)
#accuracy _ score :  0.9666666666666667

y_predict_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC :", accuracy_score(y_test, y_predict_best))

print("걸린 시간 : ", round(end_time - start_time ,2 ),"초") #걸린 시간 :  2.14 초

import pandas as pd
print(pd.DataFrame(model.cv_results_)) #화면 상 안보임. ==> csv파일로 보내서 보면 좋을것.
