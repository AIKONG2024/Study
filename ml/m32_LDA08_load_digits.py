from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import time

datasets = load_digits()
x= datasets.data
y= datasets.target

# x_train, x_test, y_train , y_test = train_test_split(
#     x, y, shuffle= True, random_state=123, train_size=0.8,
#     stratify= y
# )
random_state=42

# 모델 구성
model = RandomForestClassifier()
unique = np.unique(datasets.target)
# print(unique)
for idx in range(1,min(len(datasets.feature_names), len(unique))) :
    
    x = datasets.data
    y = datasets.target
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    lda = LinearDiscriminantAnalysis(n_components=idx)
    x = lda.fit_transform(x,y)
    
    x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
    )

    # 평가 예측
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    predict = model.predict(x_test)
    print(f'''
    lda n_components : {idx} 
    score : {accuracy_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')
evr = lda.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)
'''

        lda n_components : 1
    score : 0.3861111111111111
    걸린 시간 : 0.13 초


    lda n_components : 2
    score : 0.6805555555555556
    걸린 시간 : 0.12 초


    lda n_components : 3
    score : 0.8694444444444445
    걸린 시간 : 0.11 초


    lda n_components : 4
    score : 0.9166666666666666
    걸린 시간 : 0.14 초


    lda n_components : 5
    score : 0.9277777777777778
    걸린 시간 : 0.14 초


    lda n_components : 6
    score : 0.9472222222222222
    걸린 시간 : 0.13 초


    lda n_components : 7
    score : 0.9527777777777777
    걸린 시간 : 0.13 초


    lda n_components : 8
    score : 0.9666666666666667
    걸린 시간 : 0.13 초


    lda n_components : 9
    score : 0.9722222222222222
    걸린 시간 : 0.17 초

[0.28912041 0.18262788 0.16962345 0.1167055  0.08301253 0.06565685
 0.04310127 0.0293257  0.0208264 ]
1.0
[0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662
 0.94984789 0.9791736  1.        ]
'''