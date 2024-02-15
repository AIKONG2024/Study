from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

datasets = fetch_covtype()

#모델 구현
model = RandomForestClassifier()

for idx in range(1,len(datasets.feature_names)) :
    x = datasets.data
    y = datasets.target
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)
    #컴파일, 훈련
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
evr = pca.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

'''
[RandomForestClassifier] model acc :  0.9532081879933909
[RandomForestClassifier] eval_acc :  0.9532081879933909
    pca n_components : 1      
    score : 0.4156339898207568
    걸린 시간 : 115.67 초     
    

    pca n_components : 2      
    score : 0.4975658331489268
    걸린 시간 : 71.74 초      
    

    pca n_components : 3     
    score : 0.772983206707482
    걸린 시간 : 59.97 초     
    

    pca n_components : 4     
    score : 0.921578028570726
    걸린 시간 : 87.6 초      
    

    pca n_components : 5      
    score : 0.9483538639325317
    걸린 시간 : 90.06 초      
    

    pca n_components : 6      
    score : 0.9530316441690639
    걸린 시간 : 94.98 초 
'''