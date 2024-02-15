import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

import time

from sklearn.metrics import r2_score, mean_squared_error
path = 'C:/_data/kaggle/bike/'
train_csv =pd.read_csv(path + 'train.csv', index_col=0)
test_csv =pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

#데이터 전처리
# x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
# y = train_csv['count']
    
#데이터
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, random_state= 12345)

from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#모델 생성
model = RandomForestRegressor()
for idx in range(1,len(train_csv.columns)) :
    x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
    y = train_csv['count']
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    s_test_csv = scaler.transform(test_csv)
    
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    a_test_csv = pca.transform(s_test_csv)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, random_state= 12345)
    
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
    submit = model.predict(a_test_csv)
    # #데이터 출력
    submission_csv['count'] = submit
    import time as tm
    ltm = tm.localtime()
    file_name = f'sampleSubmission_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}.csv'
    submission_csv.to_csv(path + file_name, index = False )
    
evr = pca.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

'''
[RandomForestRegressor] mode eval_r2 :  0.26565193551130906
[RandomForestRegressor] mode eval_mse :  23736.423317567074
   pca n_components : 1
    score : 0.2688297999890691
    걸린 시간 : 0.89 초


    pca n_components : 2
    score : 0.27437400740758044
    걸린 시간 : 0.9 초


    pca n_components : 3
    score : 0.2647829408645165
    걸린 시간 : 0.93 초


    pca n_components : 4
    score : 0.26925244659796743
    걸린 시간 : 0.91 초


    pca n_components : 5
    score : 0.2662109444736871
    걸린 시간 : 0.91 초


    pca n_components : 6
    score : 0.2654218747114554
    걸린 시간 : 0.91 초


    pca n_components : 7
    score : 0.26436949514267805
    걸린 시간 : 0.97 초


    pca n_components : 8
    score : 0.26339917421274395
    걸린 시간 : 0.95 초


    pca n_components : 9
    score : 0.2705576484203881
    걸린 시간 : 0.91 초


    pca n_components : 10
    score : 0.2682317778953851
    걸린 시간 : 0.9 초
'''