import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(12)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] #(4,2)
y_data = [[0],[1], [1], [0]] #(4,1)

#m02_5번과 똑같은 레이어로 구성
#2. 모델

#레이어 1
x= tf.compat.v1.placeholder(tf.float32, shape = [None,2])
y= tf.compat.v1.placeholder(tf.float32, shape = [None,1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_uniform([2,10]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1 #(N,10)

#레이어 2
w2 = tf.compat.v1.Variable(tf.compat.v1.random_uniform([10,9]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([9]),name='bias2')
layer2 = tf.nn.relu(tf.compat.v1.matmul(layer1, w2) + b2) 

w3 = tf.compat.v1.Variable(tf.compat.v1.random_uniform([9,1]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias3')
hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(layer2, w3) + b3) 

# hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(x, w1) + b1)

#3-1 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis ) + (1-y)*tf.log(1-hypothesis ))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.2)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(1000):
    _, loss_v, w_v, b_v = sess.run([train, loss, w1, b1], feed_dict = {x:x_data, y:y_data})
    # print(loss_v, '\t', w_v, '\t', b_v)

predict = sess.run(hypothesis, feed_dict={x:x_data})
predict = np.round(predict)
print(predict)
acc = accuracy_score(y_data,predict)

print('acc : ', acc)

sess.close()