from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
SEED = 11
x,y = load_digits(return_X_y=True)

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

#1. 데이터
x,y = load_digits(return_X_y=True)

ohe = OneHotEncoder()
ohe_y = ohe.fit_transform(y.reshape(-1,1)).toarray()

#Random으로 1번만 돌리고
#Grid Search, Randomized Search 로 돌려보기
#시간체크

x_train, x_test, y_train , y_test = train_test_split(
    x, ohe_y, shuffle= True, random_state=123, train_size=0.7,
    stratify= y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape= [None,64])
y = tf.compat.v1.placeholder(tf.float32, shape= [None,10])
keep_prob = tf.compat.v1.placeholder(tf.float32)

layer1 = Layer(input_dim= 64, output_dim= 32, activation=tf.nn.relu).forward(x)
layer2 = Layer(input_dim= 32, output_dim= 16, activation=tf.nn.sigmoid).forward(layer1)
tf.compat.v1.nn.dropout(layer2, keep_prob=keep_prob)
layer3 = Layer(input_dim= 16, output_dim= 8, activation=tf.nn.sigmoid).forward(layer2)
output = Layer(input_dim= 8, output_dim= 10, activation=tf.nn.softmax).forward(layer3)

#3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(1000):
    _, loss_v = sess.run([train, loss], feed_dict = {x:x_train, y:y_train, keep_prob:0.5})
    print(loss_v, '\t')

predict = output
predict = sess.run(predict, feed_dict={x:x_test, keep_prob:1.0})
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test,predict)

print('acc : ', acc)

sess.close()

#acc :  0.9574074074074074