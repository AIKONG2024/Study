# 모델 : RanadomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import time


# 1. 데이터
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_jobs" : [-1], "n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"n_jobs" : [-1], "max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"n_jobs" : [-1], "min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"n_jobs" : [-1], "min_samples_split": [2, 3, 5, 10]},
]
rfc = RandomForestClassifier()
model = GridSearchCV(rfc, parameters, cv=kf , n_jobs=-1, refit=True, verbose=1) #cv = 5 이런식으로 넣어도 된다.
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("best_model_acc_score : ", best_acc_score) #best_acc_score :  0.9333333333333333


print(f'''
최적의 파라미터 :\t{model.best_estimator_}
최적의 매개변수 :\t{model.best_params_}
best score :\t\t{model.best_score_}
best_model_acc_score :\t{best_acc_score}
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

'''