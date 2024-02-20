from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
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
# import pickle 
# path = "C:\_data\_save\_pickle_test\\"
# model = pickle.load(open(path + "m39_pickle1_save_dat", 'rb'))
# print("pickle 불러오기 완료")

import joblib
path = "C:\_data\_save\_joblib_test\\"
model = joblib.load(path + 'm40_joblib1_save_dat')
print("joblib 불러오기 완료")

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
# ============================================
# [Best acc_score: 0.9722222222222222]
# ============================================


