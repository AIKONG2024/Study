from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import numpy as np
import time


datasets = load_wine()

from sklearn.preprocessing import StandardScaler


#모델 구현
model = RandomForestClassifier()

unique = np.unique(datasets.target)
for idx in range(1,min(len(datasets.feature_names), len(unique))) :
    x = datasets.data
    y = datasets.target
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    lda = LinearDiscriminantAnalysis(n_components=idx)
    x = lda.fit_transform(x,y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)
    #컴파일 훈련
    model.fit(x_train, y_train)

    #예측 평가
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

'''
    lda n_components : 1
    score : 0.98
    걸린 시간 : 0.05 초


    lda n_components : 2
    score : 1.0
    걸린 시간 : 0.05 초

[0.68747889 0.31252111]
1.0
'''