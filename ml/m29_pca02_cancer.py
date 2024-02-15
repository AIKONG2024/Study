import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold ,cross_val_score, StratifiedKFold, cross_val_predict, HalvingGridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV

model = RandomForestClassifier()
datasets = load_breast_cancer()
import time
for idx in range(1,len(datasets.feature_names)) :
    # 1. 데이터
    x = datasets.data
    y = datasets.target
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)

    x_train, x_test, y_train , y_test = train_test_split(
        x, y, shuffle= True, random_state=123, train_size=0.8,
        stratify= y
    )

    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    predict = model.predict(x_test)

    print(f'''
    pca n_components : {idx} 
    score : {accuracy_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')


'''
    pca n_components : 1 
    score : 0.8421052631578947
    걸린 시간 : 0.06 초
    

    pca n_components : 2 
    score : 0.9298245614035088
    걸린 시간 : 0.05 초


    pca n_components : 3
    score : 0.9210526315789473
    걸린 시간 : 0.06 초


    pca n_components : 4
    score : 0.9736842105263158
    걸린 시간 : 0.06 초


    pca n_components : 5
    score : 0.9649122807017544
    걸린 시간 : 0.06 초


    pca n_components : 6
    score : 0.9649122807017544
    걸린 시간 : 0.06 초


    pca n_components : 7
    score : 0.9736842105263158
    걸린 시간 : 0.06 초


    pca n_components : 8
    score : 0.9736842105263158
    걸린 시간 : 0.06 초


    pca n_components : 9
    score : 0.9736842105263158
    걸린 시간 : 0.07 초


    pca n_components : 10
    score : 0.9649122807017544
    걸린 시간 : 0.07 초


    pca n_components : 11
    score : 0.956140350877193
    걸린 시간 : 0.07 초


    pca n_components : 12
    score : 0.9736842105263158
    걸린 시간 : 0.07 초


    pca n_components : 13
    score : 0.9473684210526315
    걸린 시간 : 0.07 초


    pca n_components : 14
    score : 0.9649122807017544
    걸린 시간 : 0.07 초


    pca n_components : 15
    score : 0.9649122807017544
    걸린 시간 : 0.07 초


    pca n_components : 16
    score : 0.9736842105263158
    걸린 시간 : 0.08 초


    pca n_components : 17
    score : 0.9649122807017544
    걸린 시간 : 0.08 초


    pca n_components : 18
    score : 0.9736842105263158
    걸린 시간 : 0.08 초


    pca n_components : 19
    score : 0.9824561403508771
    걸린 시간 : 0.08 초


    pca n_components : 20
    score : 0.9649122807017544
    걸린 시간 : 0.09 초


    pca n_components : 21
    score : 0.9736842105263158
    걸린 시간 : 0.08 초


    pca n_components : 22
    score : 0.9736842105263158
    걸린 시간 : 0.08 초


    pca n_components : 23
    score : 0.9649122807017544
    걸린 시간 : 0.08 초


    pca n_components : 24
    score : 0.9736842105263158
    걸린 시간 : 0.08 초


    pca n_components : 25
    score : 0.9649122807017544
    걸린 시간 : 0.09 초


    pca n_components : 26
    score : 0.9649122807017544
    걸린 시간 : 0.09 초


    pca n_components : 27
    score : 0.9736842105263158
    걸린 시간 : 0.09 초


    pca n_components : 28
    score : 0.9824561403508771
    걸린 시간 : 0.09 초


    pca n_components : 29
    score : 0.9736842105263158
    걸린 시간 : 0.09 초
'''