import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
tf.set_random_seed(777)

#1. 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.compat.v1.placeholder(tf.float32, shape= [None,4])
y = tf.compat.v1.placeholder(tf.float32, shape= [None,3])

w = tf.compat.v1.Variable(tf.random_normal([4,3]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,3]), name='bias')

#2. 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

#3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(100):
    _, loss_v, w_v, b_v = sess.run([train, loss, w, b], feed_dict = {x:x_data, y:y_data})
    print(loss_v, '\t', w_v, '\t', b_v)

predict =  tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)
predict = sess.run(predict, feed_dict={x:x_data})
predict = np.argmax(predict, axis=1)
y_data = np.argmax(y_data, axis=1)
acc = accuracy_score(y_data,predict)

print('acc : ', acc)

sess.close()

# r2 :  -1.5107092582570267
# mse :  0.5325620461564587