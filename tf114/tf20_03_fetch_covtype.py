from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

class Layer:
    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.w = tf.compat.v1.Variable(tf.random.normal([input_dim, output_dim]), name='weight')
        self.b = tf.compat.v1.Variable(tf.zeros([output_dim]), name='bias')
    
    def forward(self, x):
        z = tf.matmul(x, self.w) + self.b
        if self.activation is not None:
            return self.activation(z)
        return z

x,y = fetch_covtype(return_X_y=True)

ohe = OneHotEncoder()
ohe_y = ohe.fit_transform(y.reshape(-1,1)).toarray()

x_train, x_test, y_train , y_test = train_test_split(
    x, ohe_y, shuffle= True, random_state=123, train_size=0.7,
    stratify= y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape= [None,54])
y = tf.compat.v1.placeholder(tf.float32, shape= [None,7])

#2. 모델
layer1 = Layer(input_dim= 54, output_dim= 32, activation=tf.nn.relu).forward(x)
layer2 = Layer(input_dim= 32, output_dim= 16, activation=tf.nn.sigmoid).forward(layer1)
layer3 = Layer(input_dim= 16, output_dim= 8, activation=tf.nn.sigmoid).forward(layer2)
output = Layer(input_dim= 8, output_dim= 7, activation=tf.nn.softmax).forward(layer3)

#3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.025)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(100):
    _, loss_v = sess.run([train, loss], feed_dict = {x:x_train, y:y_train})
    print(loss_v)

predict =  output
predict = sess.run(predict, feed_dict={x:x_test})
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test,predict)

print('acc : ', acc)

sess.close()
#acc :  0.7088649715439691