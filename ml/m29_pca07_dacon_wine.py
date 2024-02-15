# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time


path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

# x = train_csv.drop(columns='quality')
# y = train_csv['quality']

#데이터 분류

#모델 생성
model = RandomForestClassifier()
for idx in range(1,len(train_csv.columns)) :
    x = train_csv.drop(columns='quality')
    y = train_csv['quality']
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    s_test_csv = scaler.transform(test_csv)

    
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    a_test_csv = pca.transform(s_test_csv)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1234567, stratify=y)
    
    #컴파일 , 훈련
    model.fit(x_train, y_train)

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
    submission = model.predict(a_test_csv)
    submission_csv['quality'] = submission
    submission_csv['quality'] += 3


    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)
evr = pca.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)
'''
    pca n_components : 1
    score : 0.48363636363636364
    걸린 시간 : 0.39 초


    pca n_components : 2
    score : 0.5333333333333333
    걸린 시간 : 0.35 초


    pca n_components : 3
    score : 0.5890909090909091
    걸린 시간 : 0.35 초


    pca n_components : 4
    score : 0.6145454545454545
    걸린 시간 : 0.53 초


    pca n_components : 5
    score : 0.6218181818181818
    걸린 시간 : 0.51 초


    pca n_components : 6
    score : 0.6424242424242425
    걸린 시간 : 0.51 초


    pca n_components : 7
    score : 0.6472727272727272
    걸린 시간 : 0.5 초


    pca n_components : 8
    score : 0.6387878787878788
    걸린 시간 : 0.5 초


    pca n_components : 9
    score : 0.6424242424242425
    걸린 시간 : 0.66 초


    pca n_components : 10
    score : 0.6448484848484849
    걸린 시간 : 0.67 초


    pca n_components : 11
    score : 0.6339393939393939
    걸린 시간 : 0.67 초


    pca n_components : 12
    score : 0.64
    걸린 시간 : 0.66 초

[9.56704647e-01 3.79300330e-02 4.56924439e-03 4.58432263e-04
 2.95880132e-04 2.53319059e-05 6.16136133e-06 4.27001768e-06
 3.27971821e-06 2.48344761e-06 2.36756563e-07 8.04135589e-11]
0.9999999999999999
[0.95670465 0.99463468 0.99920392 0.99966236 0.99995824 0.99998357
 0.99998973 0.999994   0.99999728 0.99999976 1.         1.        ]
'''

