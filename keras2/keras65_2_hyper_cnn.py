#cnn 생성
#early_stopping 적용
#MCP 적용

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
x_train = x_train.reshape(60000,28,28).astype('float32')/255.
x_test = x_test.reshape(10000,28,28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation = 'relu', node1=128, node2=64, node3 = 32, lr = 0.001):
    inputs = Input(shape=(28,28,1), name='inputs')
    x = Conv2D(node1, kernel_size = 2, activation=activation, name = 'hidden1')(inputs)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Dropout(drop)(x)
    x = Conv2D(node2, kernel_size = 2, activation=activation, name = 'hidden2')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Dropout(drop)(x)
    x = Conv2D(node3, kernel_size = 2,activation=activation, name = 'hidden3')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax', name = 'outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer,metrics=['acc'], loss='sparse_categorical_crossentropy')
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    return {'batch_size':batchs, 
            'optimizer': optimizers,
            'drop':dropouts,
            'activation':activations,
            'node1': node1,
            'node2': node2,
            'node3': node3,
            }

hyperparameters = create_hyperparameter()
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
keras_model = KerasClassifier(build_model)
model = RandomizedSearchCV(keras_model, hyperparameters, cv = 3, n_iter=10, verbose=1,random_state=42)
import time 
start_time = time.time()
model.fit(x_train,y_train, epochs =50)
end_time = time.time()

print("걸린시간 : ", round(end_time - start_time,2))
print('model.best_params_ :', model.best_params_ )
print('model.best_estimator_ ', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc_score : ', accuracy_score(y_test, y_predict))

'''
걸린시간 :  492.33
model.best_params_ : {'optimizer': 'rmsprop', 'node3': 128, 'node2': 32, 'node1': 32, 'drop': 0.2, 'batch_size': 400, 'activation': 'elu'}    
model.best_estimator_  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000002B477BA8C10>
model.best_score_ : 0.9849333465099335
model.score :  0.9898999929428101
acc_score :  0.9899
'''