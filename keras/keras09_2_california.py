from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape , y.shape) #(20640, 8) (20640,)

# print(datasets.feature_names) #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# print(datasets.DESCR)
'''
:Attribute Information:
        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per household
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude
'''

#[실습]
# R2 0.55 ~ 0.6 이상
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score
    
#데이터
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=2874458)

#모델 구성
model = Sequential()
model.add(Dense(128, input_dim = 8))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=3616, batch_size=250)
end_time = time.time()


#평가, 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
result = model.predict(x)
print("loss : ", loss)
print("R2 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")


#train_size : 0.75 /  deep 6 (13-64-64-64-64-1)  / random_state : 2874458 / epochs = 3616 / batch_size = 250
# 걸린시간 :  140.17 초
# loss :  0.531786322593689
# R2 :  0.6038468068252449

# loss :  0.5676167607307434
# R2 :  0.577154799431054

# loss :  0.5420716404914856
# R2 :  0.5961846718349288
# 걸린시간 :  136.41 초
#=====================================================

#mse -> msa 
# 걸린시간 :  137.71 초
# loss :  0.5453983545303345
# R2 :  0.5117905535708138


# loss :  0.5241289734840393
# R2 :  0.5444347177914541

# loss :  0.5211647748947144
# R2 :  0.5533077586054875
# 걸린시간 :  136.91 초

# loss :  0.5457435846328735
# R2 :  0.5076286753020893
# 걸린시간 :  138.48 초
