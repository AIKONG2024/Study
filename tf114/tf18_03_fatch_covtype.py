from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

x,y = fetch_covtype(return_X_y=True)

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

x = tf.compat.v1.placeholder(tf.float32, shape= [None,54])
y = tf.compat.v1.placeholder(tf.float32, shape= [None,7])

w = tf.compat.v1.Variable(tf.random_normal([54,7]), name='weight')
b = tf.compat.v1.Variable(tf.random_normal([1,7]), name='bias')

#2. 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

#3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.2)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(100):
    _, loss_v, w_v, b_v = sess.run([train, loss, w, b], feed_dict = {x:x_train, y:y_train})
    print(loss_v, '\t', w_v, '\t', b_v)

predict =  tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)
predict = sess.run(predict, feed_dict={x:x_test})
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test,predict)

print('acc : ', acc)

sess.close()
#acc :  0.6870238204516248