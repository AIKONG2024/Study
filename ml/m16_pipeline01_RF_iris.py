# 모델 : RanadomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import time


# 1. 데이터
x,y = load_iris(return_X_y=True)

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
#     {'max_depth': 8, 'min_samples_leaf': 3, 'n_jobs': -1}
# ]
model =  make_pipeline(MinMaxScaler(), RandomForestClassifier(max_depth=8, min_samples_leaf=3, n_jobs=-1))
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# from sklearn.metrics import accuracy_score
# best_predict = model.best_estimator_.predict(x_test)
# best_acc_score = accuracy_score(y_test, best_predict)

# print("best_model_acc_score : ", best_acc_score) #best_acc_score :  0.9333333333333333


# print(f'''
# 최적의 파라미터 :\t{model.best_estimator_}
# 최적의 매개변수 :\t{model.best_params_}
# best score :\t\t{model.best_score_}
# best_model_acc_score :\t{best_acc_score}
# 걸린 시간 : {round(end_time - start_time ,2 )} 초
# ''')
predict = model.predict(x_test)

print(f'''
score : {accuracy_score(y_test, predict)}
걸린 시간 : {round(end_time - start_time ,2 )} 초
''')


'''
Fitting 5 folds for each of 48 candidates, totalling 240 fits
best_model_acc_score :  0.9333333333333333

최적의 파라미터 :       RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_jobs=-1)
최적의 매개변수 :       {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100, 'n_jobs': -1}
best score :            0.9583333333333334
best_model_acc_score :  0.9333333333333333
걸린 시간 : 3.78 초

================
RandomizedSearchCV
Fitting 5 folds for each of 10 candidates, totalling 50 fits
best_model_acc_score :  0.9666666666666667

최적의 파라미터 :       RandomForestClassifier(max_depth=8, min_samples_leaf=5, n_jobs=-1)
최적의 매개변수 :       {'n_jobs': -1, 'min_samples_leaf': 5, 'max_depth': 8}
best score :            0.9583333333333334
best_model_acc_score :  0.9666666666666667
걸린 시간 : 2.45 초

==================
halving
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 10
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 48
n_resources: 10
Fitting 5 folds for each of 48 candidates, totalling 240 fits
----------
iter: 1
n_candidates: 16
n_resources: 30
Fitting 5 folds for each of 16 candidates, totalling 80 fits
----------
iter: 2
n_candidates: 6
n_resources: 90
Fitting 5 folds for each of 6 candidates, totalling 30 fits
best_model_acc_score :  0.9666666666666667

최적의 파라미터 :       RandomForestClassifier(max_depth=8, min_samples_leaf=3, n_jobs=-1)
최적의 매개변수 :       {'max_depth': 8, 'min_samples_leaf': 3, 'n_jobs': -1}
best score :            0.9444444444444443
best_model_acc_score :  0.9666666666666667
걸린 시간 : 5.34 초

====================
pipeline

score : 0.9333333333333333
걸린 시간 : 0.06 초
'''