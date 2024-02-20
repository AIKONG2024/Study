from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
import xgboost as xgb
seed = 42

datasets = load_digits()
x= datasets.data
y= datasets.target

X_train, X_test, y_train , y_test = train_test_split(
x, y, shuffle= True, random_state=123, train_size=0.8,
stratify= y
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {
    'n_estimators': 4000,
    'learning_rate': 0.01,  
    'max_depth': 2, 
    'early_stopping_rounds' : 100
}

# 2. 모델
model = XGBClassifier(random_state = seed, **parameters)
model.fit(X_train, y_train, eval_set = [(X_test, y_test)], eval_metric = "merror")
#분류 : mlogloss, merror, auc (이진이 더 좋음)  // f1 score: mlogloss
#문서 : https://xgboost.readthedocs.io/en/stable/parameter.html
print("파라미터 : ",model.get_params())

#4. 평가 예측
results = model.score(X_test, y_test)
x_predictsion = model.predict(X_test)
best_score = accuracy_score(y_test, x_predictsion) 
print(
f"""
============================================
[Best acc_score: {best_score}]
============================================
"""
)
################################################
# import pickle 

# path = "C:\_data\_save\_pickle_test\\"
# pickle.dump(model, open(path + "m39_pickle1_save.dat", 'wb'))
# print("pickle 저장완료")


# import joblib
# path = "C:\_data\_save\_joblib_test\\"
# joblib.dump(model, path + 'm40_joblib1_save.dat')
# print("joblib 저장완료")

path = "C:\_data\_save\\"
model.save_model(path + 'm41_xgb1_save_model.json')
print("xgb 저장 완료")
