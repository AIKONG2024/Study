from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
for idx in range(1,len(datasets.feature_names)) :
    
    x = datasets.data
    y = datasets.target
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    
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
    pca n_components : {idx} 
    score : {accuracy_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')
evr = pca.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)
'''

    pca n_components : 1
    score : 0.2916666666666667
    걸린 시간 : 0.13 초


    pca n_components : 2
    score : 0.5416666666666666
    걸린 시간 : 0.13 초


    pca n_components : 3
    score : 0.7638888888888888
    걸린 시간 : 0.12 초


    pca n_components : 4
    score : 0.8583333333333333
    걸린 시간 : 0.15 초


    pca n_components : 5
    score : 0.8861111111111111
    걸린 시간 : 0.15 초


    pca n_components : 6
    score : 0.9
    걸린 시간 : 0.16 초


    pca n_components : 7
    score : 0.925
    걸린 시간 : 0.16 초


    pca n_components : 8
    score : 0.9416666666666667
    걸린 시간 : 0.16 초


    pca n_components : 9
    score : 0.9388888888888889
    걸린 시간 : 0.19 초


    pca n_components : 10
    score : 0.9361111111111111
    걸린 시간 : 0.19 초


    pca n_components : 11
    score : 0.9472222222222222
    걸린 시간 : 0.19 초


    pca n_components : 12
    score : 0.9444444444444444
    걸린 시간 : 0.19 초


    pca n_components : 13
    score : 0.9416666666666667
    걸린 시간 : 0.19 초


    pca n_components : 14
    score : 0.9527777777777777
    걸린 시간 : 0.19 초


    pca n_components : 15
    score : 0.9638888888888889
    걸린 시간 : 0.19 초


    pca n_components : 16
    score : 0.9638888888888889
    걸린 시간 : 0.23 초


    pca n_components : 17
    score : 0.9666666666666667
    걸린 시간 : 0.23 초


    pca n_components : 18
    score : 0.9666666666666667
    걸린 시간 : 0.22 초


    pca n_components : 19
    score : 0.9777777777777777
    걸린 시간 : 0.23 초


    pca n_components : 20
    score : 0.9694444444444444
    걸린 시간 : 0.23 초


    pca n_components : 21
    score : 0.9722222222222222
    걸린 시간 : 0.23 초


    pca n_components : 22
    score : 0.975
    걸린 시간 : 0.23 초


    pca n_components : 23
    score : 0.9638888888888889
    걸린 시간 : 0.23 초


    pca n_components : 24
    score : 0.9694444444444444
    걸린 시간 : 0.23 초


    pca n_components : 25
    score : 0.9611111111111111
    걸린 시간 : 0.26 초


    pca n_components : 26
    score : 0.9583333333333334
    걸린 시간 : 0.27 초


    pca n_components : 27
    score : 0.9638888888888889
    걸린 시간 : 0.26 초


    pca n_components : 28
    score : 0.9777777777777777
    걸린 시간 : 0.26 초


    pca n_components : 29
    score : 0.9666666666666667
    걸린 시간 : 0.26 초


    pca n_components : 30
    score : 0.9666666666666667
    걸린 시간 : 0.26 초


    pca n_components : 31
    score : 0.9722222222222222
    걸린 시간 : 0.26 초


    pca n_components : 32
    score : 0.9694444444444444
    걸린 시간 : 0.26 초


    pca n_components : 33
    score : 0.9722222222222222
    걸린 시간 : 0.28 초


    pca n_components : 34
    score : 0.9694444444444444
    걸린 시간 : 0.27 초


    pca n_components : 35
    score : 0.9638888888888889
    걸린 시간 : 0.27 초


    pca n_components : 36
    score : 0.9694444444444444
    걸린 시간 : 0.3 초


    pca n_components : 37
    score : 0.9722222222222222
    걸린 시간 : 0.31 초


    pca n_components : 38
    score : 0.9722222222222222
    걸린 시간 : 0.3 초


    pca n_components : 39
    score : 0.9722222222222222
    걸린 시간 : 0.3 초


    pca n_components : 40
    score : 0.9722222222222222
    걸린 시간 : 0.31 초


    pca n_components : 41
    score : 0.9805555555555555
    걸린 시간 : 0.31 초


    pca n_components : 42
    score : 0.975
    걸린 시간 : 0.3 초


    pca n_components : 43
    score : 0.9722222222222222
    걸린 시간 : 0.3 초


    pca n_components : 44
    score : 0.9694444444444444
    걸린 시간 : 0.3 초


    pca n_components : 45
    score : 0.975
    걸린 시간 : 0.31 초


    pca n_components : 46
    score : 0.9722222222222222
    걸린 시간 : 0.31 초


    pca n_components : 47
    score : 0.9722222222222222
    걸린 시간 : 0.31 초


    pca n_components : 48
    score : 0.9694444444444444
    걸린 시간 : 0.31 초


    pca n_components : 49
    score : 0.9722222222222222
    걸린 시간 : 0.35 초


    pca n_components : 50
    score : 0.9722222222222222
    걸린 시간 : 0.35 초


    pca n_components : 51
    score : 0.9694444444444444
    걸린 시간 : 0.34 초


    pca n_components : 52
    score : 0.9722222222222222
    걸린 시간 : 0.34 초


    pca n_components : 53
    score : 0.9833333333333333
    걸린 시간 : 0.34 초


    pca n_components : 54
    score : 0.975
    걸린 시간 : 0.36 초


    pca n_components : 55
    score : 0.9805555555555555
    걸린 시간 : 0.34 초


    pca n_components : 56
    score : 0.9638888888888889
    걸린 시간 : 0.35 초


    pca n_components : 57
    score : 0.9722222222222222
    걸린 시간 : 0.35 초


    pca n_components : 58
    score : 0.975
    걸린 시간 : 0.36 초


    pca n_components : 59
    score : 0.975
    걸린 시간 : 0.35 초


    pca n_components : 60
    score : 0.9777777777777777
    걸린 시간 : 0.35 초


    pca n_components : 61
    score : 0.9722222222222222
    걸린 시간 : 0.36 초


    pca n_components : 62
    score : 0.9555555555555556
    걸린 시간 : 0.35 초


    pca n_components : 63
    score : 0.9694444444444444
    걸린 시간 : 0.34 초

[1.20339161e-01 9.56105440e-02 8.44441489e-02 6.49840791e-02
 4.86015488e-02 4.21411987e-02 3.94208280e-02 3.38938092e-02
 2.99822101e-02 2.93200255e-02 2.78180546e-02 2.57705509e-02
 2.27530332e-02 2.22717974e-02 2.16522943e-02 1.91416661e-02
 1.77554709e-02 1.63806927e-02 1.59646017e-02 1.48919119e-02
 1.34796957e-02 1.27193137e-02 1.16583735e-02 1.05764660e-02
 9.75315947e-03 9.44558990e-03 8.63013827e-03 8.36642854e-03
 7.97693248e-03 7.46471371e-03 7.25582151e-03 6.91911245e-03
 6.53908536e-03 6.40792574e-03 5.91384112e-03 5.71162405e-03
 5.23636803e-03 4.81807586e-03 4.53719260e-03 4.23162753e-03
 4.06053070e-03 3.97084808e-03 3.56493303e-03 3.40787181e-03
 3.27835335e-03 3.11032007e-03 2.88575294e-03 2.76489264e-03
 2.59174941e-03 2.34483006e-03 2.18256858e-03 2.03597635e-03
 1.95512426e-03 1.83318499e-03 1.67946387e-03 1.61236062e-03
 1.47762694e-03 1.35118411e-03 1.25100742e-03 1.03695730e-03
 8.25350945e-04 3.23475858e-33 6.39352227e-34]
1.0
[0.12033916 0.21594971 0.30039385 0.36537793 0.41397948 0.45612068
 0.49554151 0.52943532 0.55941753 0.58873755 0.61655561 0.64232616
 0.66507919 0.68735099 0.70900328 0.72814495 0.74590042 0.76228111
 0.77824572 0.79313763 0.80661732 0.81933664 0.83099501 0.84157148
 0.85132464 0.86077023 0.86940036 0.87776679 0.88574372 0.89320844
 0.90046426 0.90738337 0.91392246 0.92033038 0.92624422 0.93195585
 0.93719222 0.94201029 0.94654748 0.95077911 0.95483964 0.95881049
 0.96237542 0.9657833  0.96906165 0.97217197 0.97505772 0.97782262
 0.98041436 0.98275919 0.98494176 0.98697774 0.98893286 0.99076605
 0.99244551 0.99405787 0.9955355  0.99688668 0.99813769 0.99917465
 1.         1.         1.        ]
'''