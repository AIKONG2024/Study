from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import time


datasets = load_wine()

from sklearn.preprocessing import StandardScaler


#모델 구현
model = RandomForestClassifier()

for idx in range(1,len(datasets.feature_names)) :
    x = datasets.data
    y = datasets.target
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)
    #컴파일 훈련
    model.fit(x_train, y_train)

    #예측 평가
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
    score : 0.78
    걸린 시간 : 0.05 초


    pca n_components : 2
    score : 0.82
    걸린 시간 : 0.05 초


    pca n_components : 3
    score : 0.92
    걸린 시간 : 0.05 초


    pca n_components : 4
    score : 0.94
    걸린 시간 : 0.05 초


    pca n_components : 5
    score : 0.96
    걸린 시간 : 0.05 초


    pca n_components : 6
    score : 0.96
    걸린 시간 : 0.05 초


    pca n_components : 7
    score : 0.94
    걸린 시간 : 0.05 초


    pca n_components : 8
    score : 0.98
    걸린 시간 : 0.05 초


    pca n_components : 9
    score : 0.98
    걸린 시간 : 0.05 초


    pca n_components : 10
    score : 0.96
    걸린 시간 : 0.05 초


    pca n_components : 11
    score : 0.96
    걸린 시간 : 0.05 초


    pca n_components : 12
    score : 0.96
    걸린 시간 : 0.05 초
'''