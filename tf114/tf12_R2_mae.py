import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]
x_test = [4,5,6]
y_test = [4,5,6]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([0.1], dtype=tf.float32, name='weight')
b = tf.compat.v1.Variable([0], dtype=tf.float32, name='weight')

#2. 모델
hypothesis = x * w #순전파

#3-1. 컴파일 // model compile(loss='mse', optimizer = 'sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse
# loss = tf.reduce_mean(tf.sqrt(hypothesis - y)) #mae

#######################옵티마이저######################
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)
# train = optimizer.minimize(loss)
lr = 0.1
gradient = tf.reduce_mean((x * w  - y) * x) #역전파
descent = w - lr * gradient
update = w.assign(descent) #<<===여기까지가 경사하강법
######################################################

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(200):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict = {x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)

from sklearn.metrics import r2_score, mean_absolute_error
prediction = x_test * w_v
r2 = r2_score(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
print('r2' , r2 )
print('mae' , mae )