#https://dacon.io/competitions/open/236070/
import numpy as np
import pandas as pd
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


path = "C:/_data/dacon/iris/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


#모델 생성
model = RandomForestClassifier()
unique = np.unique(train_csv['species'])
for idx in range(1,min(len(test_csv.columns), len(unique))) :
    x = train_csv.drop(columns='species')
    y = train_csv['species']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    s_test_csv = scaler.transform(test_csv)
    
    lda = LinearDiscriminantAnalysis(n_components=idx)
    x = lda.fit_transform(x,y)
    a_test_csv = lda.transform(s_test_csv)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=200, stratify=y)
    
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
evr = lda.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)
'''
    lda n_components : 1
    score : 0.9722222222222222
    걸린 시간 : 0.05 초


    lda n_components : 2
    score : 1.0
    걸린 시간 : 0.05 초

[0.99229795 0.00770205]
1.0
[0.99229795 1.        ]
'''
