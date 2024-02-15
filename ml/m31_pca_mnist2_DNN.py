from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
x = np.concatenate([x_train, x_test], axis=0) #(70000, 28, 28)
print(x.shape) #(70000, 28, 28)

############실습2###############
#4가지 모델을 만들어
# input_shape = ()
# 1.70000,154,
# 2.70000,331
# 3.70000,486
# 4.70000,713
# 5.70000,784
#시간과 성능 체크
#결과1. PCA = 154
#걸린시간 0000초
#acc = 0.0000
####################################
x = x.reshape(-1, 28*28)
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

components = [154, 331,486,713,784] 

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
for component in components : 
    pca = PCA(n_components=component)
    pca_x = pca.fit_transform(x)
    
    x_train = pca_x[:60000]
    x_test = pca_x[60000:]
        
    #모델구성
    model = Sequential()
    model.add(Dense(256, input_shape = (x_train.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    #컴파일 훈련
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    start_time = time.time()
    model.fit(x_train, y_train, epochs=50, batch_size=300, validation_split=0.3, callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", patience=100, restore_best_weights=True)
    ], verbose=0)
    end_time = time.time()
    
    loss = model.evaluate(x_test, y_test, verbose=0)
    predict = np.argmax(model.predict(x_test, verbose=0), axis=1) 
    print(f"PCA = {component}")
    print("acc score :", accuracy_score(y_test,predict))
    print("걸린시간 : ", round(end_time - start_time ,2 ), "초")
    
#시간과 성능 체크
#결과1. PCA = 154
#걸린시간 0000초
#acc = 0.0000

'''
PCA = 154
acc score : 0.9666
걸린시간 :  8.88 초
PCA = 331
acc score : 0.9643
걸린시간 :  8.93 초
PCA = 486
acc score : 0.9653
걸린시간 :  9.01 초
PCA = 713
acc score : 0.9644
걸린시간 :  9.23 초
PCA = 784
acc score : 0.9633
걸린시간 :  9.27 초
'''
