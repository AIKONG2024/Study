from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
x = np.concatenate([x_train, x_test], axis=0) #(70000, 28, 28)
y = np.concatenate([y_train, y_test], axis=0) #(70000, 28, 28)
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

unique = np.unique(y_train)
# print(unique)
for idx in range(1,min(28*28, len(unique))) :
    lda = LinearDiscriminantAnalysis(n_components=idx)
    lda_x = lda.fit_transform(x,y)
    
    x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
    )
        
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
    print(f"LDA = {idx}")
    print("acc score :", accuracy_score(y_test,predict))
    print("걸린시간 : ", round(end_time - start_time ,2 ), "초")
    
#시간과 성능 체크
#결과1. PCA = 154
#걸린시간 0000초
#acc = 0.0000

'''
LDA = 1
acc score : 0.9474285714285714
걸린시간 :  9.38 초
LDA = 2
acc score : 0.9450714285714286
걸린시간 :  9.46 초
LDA = 3
acc score : 0.9397142857142857
걸린시간 :  9.45 초
LDA = 4
acc score : 0.9461428571428572
걸린시간 :  8.96 초
LDA = 5
acc score : 0.9487142857142857
걸린시간 :  9.34 초
LDA = 6
acc score : 0.9500714285714286
걸린시간 :  8.73 초
LDA = 7
acc score : 0.953
걸린시간 :  8.96 초
LDA = 8
acc score : 0.9512857142857143
걸린시간 :  8.48 초
LDA = 9
acc score : 0.9411428571428572
걸린시간 :  8.68 초
========================


'''
