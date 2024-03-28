#7load diabetes
#8california
#9dacon 따릉
#10kaggle bike

import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

#1.데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)#(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=777)
x_p = tf.compat.v1.placeholder(tf.float32, shape=[None,10])
y_p = tf.compat.v1.placeholder(tf.float32, shape=[None,])
print(y_train.shape) #(353,)
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], name='weights'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias'))

#2. 모델
hypothesis  = tf.compat.v1.matmul(x_p,w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_p)) #mse

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())
#3-2. 훈련
for step in range(100):
    _, loss_v = sess.run([train, loss], feed_dict={x_p: x_train, y_p: y_train})
    print(step, '\t', loss_v, '\t')

# 예측
pred = tf.compat.v1.matmul(x_p,w) + b
predict_val = sess.run(pred, feed_dict={x_p: x_test})
# 평가
r2 = r2_score(y_test, predict_val)
mae = mean_absolute_error(y_test, predict_val)
print('r2:', r2)
print('mae:', mae)
'''
r2: -4.624809698929435
mae: 159.14129374025578
'''