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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
unique = np.unique(train_csv['quality'])
for idx in range(1,min(len(test_csv.columns), len(unique))) :
    x = train_csv.drop(columns='quality')
    y = train_csv['quality']
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    s_test_csv = scaler.transform(test_csv)

    
    lda = LinearDiscriminantAnalysis(n_components=idx)
    x = lda.fit_transform(x,y)
    a_test_csv = lda.transform(s_test_csv)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1234567, stratify=y)
    
    #컴파일 , 훈련
    model.fit(x_train, y_train)

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
    submission = model.predict(a_test_csv)
    submission_csv['quality'] = submission
    submission_csv['quality'] += 3


    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)
evr = lda.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)
'''

    lda n_components : 1
    score : 0.5709090909090909
    걸린 시간 : 0.37 초


    lda n_components : 2
    score : 0.5963636363636363
    걸린 시간 : 0.36 초


    lda n_components : 3
    score : 0.6278787878787879
    걸린 시간 : 0.36 초


    lda n_components : 4
    score : 0.6303030303030303
    걸린 시간 : 0.54 초


    lda n_components : 5
    score : 0.6327272727272727
    걸린 시간 : 0.53 초


    lda n_components : 6
    score : 0.6315151515151515
    걸린 시간 : 0.53 초

[0.85094868 0.09207847 0.03643252 0.01113301 0.00689485 0.00251248]
0.9999999999999999
[0.85094868 0.94302714 0.97945966 0.99059267 0.99748752 1.        ]
'''

