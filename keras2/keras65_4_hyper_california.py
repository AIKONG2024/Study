#회귀
import numpy as np
from sklearn.datasets import fetch_california_housing
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#1. Data
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. Model
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, callbacks=None):
    inputs = Input(shape=(8,)) 
    x = Dense(node1, activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation)(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, activation='linear')(x)  
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mse')
    if callbacks:
        model.fit(x_train, y_train, callbacks=callbacks)
    return model

# Hyperparameters
def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    return {'batch_size': batchs,
            'optimizer': optimizers,
            'drop': dropouts,
            'activation': activations,
            'node1': node1,
            'node2': node2,
            'node3': node3}

hyperparameters = create_hyperparameter()

keras_model = KerasRegressor(build_fn=build_model, verbose=1)

early_stopping = EarlyStopping(monitor='loss', mode='min', patience=40, restore_best_weights=True)
mcp = ModelCheckpoint(filepath='C:/_data/_save/keras65_save_model4.h5', monitor='loss', mode='min', save_best_only=True)

model = RandomizedSearchCV(estimator=keras_model, param_distributions=hyperparameters, cv=3, n_iter=10, verbose=1, random_state=42)

fit_params = {'callbacks': [early_stopping, mcp], 'batch_size' : 1024}
import time
start_time = time.time()
model.fit(x_train, y_train, **fit_params, epochs = 100)
end_time = time.time()
print("걸린시간 : ", round(end_time - start_time,2))
print('model.best_params_ :', model.best_params_ )
print('model.best_estimator_ ', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score : ', model.score(x_test, y_test))
predict = model.best_estimator_.predict(x_test)
print('model.score : ', r2_score(predict, y_test))
print('mse loss : ', mean_squared_error(predict, y_test))
'''
걸린시간 :  88.05
model.best_params_ : {'optimizer': 'rmsprop', 'node3': 128, 'node2': 32, 'node1': 32, 'drop': 0.2, 'batch_size': 400, 'activation': 'elu'}
model.best_estimator_  <keras.wrappers.scikit_learn.KerasRegressor object at 0x000001B526B55F10>
model.best_score_ : -0.3716195225715637
11/11 [==============================] - 0s 2ms/step - loss: 0.3786
model.score :  -0.3785589933395386
11/11 [==============================] - 0s 593us/step
model.score :  0.6327407305435993
mse loss :  0.3785590148582157
'''