import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] #(4,2)
y_data = [[0],[1], [1], [0]] #(4,1)

x= tf.compat.v1.placeholder(tf.float32, shape = [None,2])
y= tf.compat.v1.placeholder(tf.float32, shape = [None,1])
w = tf.compat.v1.Variable(tf.random_uniform([2,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1]),name='bias')

#2. 모델
hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(x, w) + b)

#3-1 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis ) + (1-y)*tf.log(1-hypothesis ))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.012)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(100):
    _, loss_v, w_v, b_v = sess.run([train, loss, w, b], feed_dict = {x:x_data, y:y_data})
    print(loss_v, '\t', w_v, '\t', b_v)

predict =  tf.nn.sigmoid(tf.compat.v1.matmul(x, w) + b)
predict = sess.run(predict, feed_dict={x:x_data})
predict = np.round(predict)
acc = accuracy_score(y_data,predict)

print('acc : ', acc)

sess.close()