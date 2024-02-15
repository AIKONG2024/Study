import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold ,cross_val_score, StratifiedKFold, cross_val_predict, HalvingGridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV

model = RandomForestClassifier()
datasets = load_breast_cancer()
import time

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

    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    predict = model.predict(x_test)

    print(f'''
    pca n_components : {idx} 
    score : {accuracy_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')
    
evr = lda.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)


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
    
[9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
 8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
 3.44135279e-07 1.86018721e-07 6.99473205e-08 1.65908880e-08
 6.99641650e-09 4.78318306e-09 2.93549214e-09 1.41684927e-09
 8.29577731e-10 5.20405883e-10 4.08463983e-10 3.63313378e-10
 1.72849737e-10 1.27487508e-10 7.72682973e-11 6.28357718e-11
 3.57302295e-11 2.76396041e-11 8.14452259e-12 6.30211541e-12
 4.43666945e-12]
0.9999999999984466
[0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
 0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
 0.99999999 0.99999999 1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.        ]
'''