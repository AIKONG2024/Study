import numpy as np
from sklearn.datasets import load_breast_cancer
import tensorflow as tf

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR) #설명  , 평균 등
print(datasets.feature_names) #컬럼명

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(569, 30) (569,)
import pandas as pd

#1
unique, counts = np.unique(y, return_counts = True)#[0 1] [212 357]
print(unique, counts)
#2
print(pd.value_counts(y))
print(pd.Series(y).value_counts())
#3
y_df = pd.DataFrame(y)
print(y_df[y_df[0] == 0].count())
print(y_df[y_df[0] == 1].count())


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234, stratify=y)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim = 30)) #기본 함수는 linear
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid')) #이진함수에서는 sigmoid 는 최종레이어에서 들어가야함.

#컴파일 , 훈련
from keras.optimizers import Adam
learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
for learning_rate in learning_rates : 
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate),
                metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs= 200, batch_size=1, 
            validation_split=0.3,verbose=0)

    #평가 예측
    loss = model.evaluate(x_test, y_test)
    y_predict = np.round(model.predict(x_test))
    # y_predict = model.predict(x_test)

    print(y_predict)

    #mse, rmse , rmsle, r2
    import numpy as np
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_test, y_predict)
    print(f"accuracy : {acc}")
    print(loss)


    print("lr : {0}, 로스 : {1}".format(learning_rate, loss))
    print("lr : {0}, ACC : {1}".format(learning_rate, acc))

'''
lr : 1.0, 로스 : [2762182400.0, 0.9824561476707458]
lr : 1.0, ACC : 0.9824561403508771

lr : 0.1, 로스 : [109609.78125, 0.9239766001701355]
lr : 0.1, ACC : 0.9239766081871345

lr : 0.01, 로스 : [0.2030820995569229, 0.9590643048286438]
lr : 0.01, ACC : 0.9590643274853801

lr : 0.001, 로스 : [0.31864863634109497, 0.9532163739204407]
lr : 0.001, ACC : 0.9532163742690059

lr : 0.0001, 로스 : [0.12497778981924057, 0.9649122953414917]
lr : 0.0001, ACC : 0.9649122807017544

=============ep 200
lr : 1.0, 로스 : [247889051648.0, 0.9824561476707458]
lr : 1.0, ACC : 0.9824561403508771

lr : 0.1, 로스 : [342938944.0, 0.9590643048286438]
lr : 0.1, ACC : 0.9590643274853801

lr : 0.01, 로스 : [342938944.0, 0.9590643048286438]
lr : 0.01, ACC : 0.9590643274853801

lr : 0.001, 로스 : [342938944.0, 0.9590643048286438]
lr : 0.001, ACC : 0.9590643274853801

lr : 0.0001, 로스 : [342938944.0, 0.9590643048286438]
lr : 0.0001, ACC : 0.9590643274853801
'''
