from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import time

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, random_state=2874458)

# 모델 구성
model = RandomForestRegressor()
for idx in range(1,len(datasets.feature_names)) :

    # 컴파일, 훈련
    model.fit(x_train, y_train)

    # 평가 예측
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    predict = model.predict(x_test)
    print(f'''
    pca n_components : {idx} 
    score : {r2_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')

'''
[RandomForestRegressor] model r2 :  0.8106541738490985
[RandomForestRegressor] mode eval_r2 :  0.8106541738490985

    pca n_components : 1
    score : 0.8076947845788855
    걸린 시간 : 5.2 초


    pca n_components : 2 
    score : 0.808925788077648
    걸린 시간 : 5.22 초
    

    pca n_components : 3 
    score : 0.8089196599251852
    걸린 시간 : 5.23 초


    pca n_components : 4
    score : 0.8100971725370383
    걸린 시간 : 5.29 초


    pca n_components : 5
    score : 0.809188765969756
    걸린 시간 : 5.31 초


    pca n_components : 6
    score : 0.8088622201953869
    걸린 시간 : 5.26 초


    pca n_components : 7
    score : 0.8080221898726034
    걸린 시간 : 5.16 초
'''

