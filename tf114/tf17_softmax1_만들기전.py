import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
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
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y)) #mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(100):
    _, loss_v, w_v, b_v = sess.run([train, loss, w, b], feed_dict = {x:x_data, y:y_data})

predict = tf.compat.v1.matmul(x,w) + b
predict = sess.run(predict, feed_dict={x:x_data})
print(predict)
r2 = r2_score(y_data,predict)
mean_squared_error = mean_squared_error(y_data,predict)

print('r2 : ', r2)
print('mse : ', mean_squared_error)

sess.close()


# #3-2 
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# #3-2. 훈련
# for step in range(100):
#     _, loss_v, w_v, b_v = sess.run([train, loss, w, b], feed_dict = {x:x_data, y:y_data})
#     print(step, '\t', loss_v, '\t')

# #4. 평가 및 예측
# predict = tf.matmul(x, w) + b  
# predict = sess.run(predict, feed_dict={x: x_data})
# r2 = r2_score(y_data, predict)
# mse = mean_squared_error(y_data, predict)
# print('r2' , r2 ) 
# print('mse' , mse ) 

# sess.close()
# '''
# r2 -134.21639530747663
# mse 28.178413434673093
# '''
