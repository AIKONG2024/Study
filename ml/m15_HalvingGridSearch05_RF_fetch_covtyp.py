import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, HalvingGridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

x,y = fetch_covtype(return_X_y=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
]
rfc = RandomForestClassifier()
model = HalvingGridSearchCV(rfc, parameters, cv=kf , n_jobs=-1, refit=True, verbose=1, random_state=42, factor=3)
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
최적의 파라미터 :       RandomForestClassifier(min_samples_split=3, n_jobs=2)
최적의 매개변수 :       {'min_samples_split': 3, 'n_jobs': 2}
best score :            0.7961965340429675
best_model_acc_score :  0.7985876732956021

Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간 : 521.09 초
best_model_acc_score :  0.9556121614760377

최적의 파라미터 :       RandomForestClassifier(n_jobs=-1)
최적의 매개변수 :       {'n_jobs': -1, 'min_samples_split': 2}
best score :            0.9499385766144254
best_model_acc_score :  0.9556121614760377
================
rdsearch
걸린 시간 : 2962.69 초
best_model_acc_score :  0.9557412459230828

최적의 파라미터 :                                                                            
RandomForestClassifier()
최적의 매개변수 :                                                                            
{'min_samples_split': 2}
best score :                                                                                
0.950353800293626
best_model_acc_score :                                                                       
0.9557412459230828

'''