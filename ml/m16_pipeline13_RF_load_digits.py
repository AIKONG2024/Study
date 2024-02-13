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
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
# n_splits = 5
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# parameters = [
#     {"n_jobs" : [-1], "n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
#     {"n_jobs" : [-1], "max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
#     {"n_jobs" : [-1], "min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
#     {"n_jobs" : [-1], "min_samples_split": [2, 3, 5, 10]},
# ]

# # 2. 모델
# model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv=kf, verbose=1, refit=True, factor=3)
# model = RandomForestClassifier()
from sklearn.pipeline import make_pipeline
model = make_pipeline(MinMaxScaler(), RandomForestClassifier(min_samples_split= 2))
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# print("최적의 매개변수 ", model.best_estimator_)
# print("최적의 파라미터 ", model.best_params_)
# print("best score : ", model.best_score_) 

# y_predict = model.predict(x_test)
# acc_score = accuracy_score(y_test, y_predict)
# print("accuracy _ score : ", acc_score)

# y_predict_best = model.best_estimator_.predict(x_test)
# print("최적 튠 ACC :", accuracy_score(y_test, y_predict_best))
# # y_predict = model.predict(x_test)
# # print("ACC :", accuracy_score(y_test, y_predict))

# print("걸린 시간 : ", round(end_time - start_time ,2 ),"초") #걸린 시간 :  2.14 초
from sklearn.metrics import accuracy_score
predict = model.predict(x_test)

print(f'''
score : {accuracy_score(y_test, predict)}
걸린 시간 : {round(end_time - start_time ,2 )} 초
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
'''