import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#1.데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]] #(6,2)
y_data = [[0],[0],[0],[1],[1],[1]]              #(6,1)

###############################
###[실습]
###############################
x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]),name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')

#2. 모델
hypothesis = tf.matmul(x, w) + b

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
for step in range(100):
    _, loss_v, w_v, b_v = sess.run([train, loss, w, b], feed_dict = {x:x_data, y:y_data})
    print(step, '\t', loss_v, '\t')

# print(w_v) 
# [[0.7726711 ]
#  [0.66013587]]
# print(type(w_v)) #<class 'numpy.ndarray'> sess 를 통과해서 나온 결과는 numpy 임.

#4. 평가 예측
x_test = tf.compat.v1.placeholder(tf.float32, shape = [None,2])
prediction = tf.matmul(x_test,w_v) + b_v
prediction = sess.run(prediction, feed_dict={x_test:x_data})
print(prediction)

#4. 평가 및 예측
predict = tf.matmul(x, w) + b  
predict = sess.run(predict, feed_dict={x: x_data})
r2 = r2_score(y_data, predict)
mse = mean_squared_error(y_data, predict)
print('r2' , r2 ) #r2 -58.95925757488294
print('mse' , mse ) #mse 14.989814393720735

sess.close()