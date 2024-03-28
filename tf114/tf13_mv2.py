import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(777)

#1. 데이터
x_data = [[73,51,65],
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]] #(5,3)
y_data = [[152],[185],[180],[205],[142]] #(5,1)
x = tf.compat.v1.placeholder(tf.float32, shape=[None,3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]),name='weight') #행렬의 곱이기 때문에 곱셈에 적합한 형태를 맞춰줘야함. n,3 * 3* 1 = n,1
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name='bias') #행렬의 덧셈이기 때문에 크기는 상관없음

#2. 모델
hypothesis = tf.matmul(x, w) + b

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    _, loss_v, w_v, b_v = sess.run([train, loss, w, b], feed_dict = {x:x_data, y:y_data})
    print(step, '\t', loss_v, '\t')
from sklearn.metrics import r2_score, mean_absolute_error
predict = tf.matmul(x, w) + b 
predict = sess.run(predict, feed_dict={x: x_data})
r2 = r2_score(y_data, predict)
mae = mean_absolute_error(y_data, predict)
print('r2' , r2 )
print('mae' , mae )