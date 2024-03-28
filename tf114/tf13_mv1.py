import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

#1. 데이터
x1_data = [73., 93., 89., 96., 73.] #국
x2_data = [80., 88., 91., 98., 66.] #영
x3_data = [75., 93., 90., 100., 70.] #수
y_data = [152., 185., 180., 196., 142.] #환산점수

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight1')
w2 = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight2')
w3 = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight3')
b = tf.compat.v1.Variable([10], dtype=tf.float32, name='bias')

#2. 모델
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

#3-1. 컴파일 // model compile(loss='mse', optimizer = 'sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse
# loss = tf.reduce_mean(tf.sqrt(hypothesis - y)) #mae

#######################옵티마이저######################
learning_rate = 0.00003
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
# lr = 0.1
# gradient = tf.reduce_mean((x * w  - y) * x)
# descent = w - lr * gradient
# update = w.assign(descent) #<<===여기까지가 경사하강법
######################################################

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_v, w1_v, w2_v, w3_v, b_v = sess.run([train,loss, w1, w2, w3, b], feed_dict = {x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    print(step, '\t', loss_v, '\t', w1_v, '\t', w2_v, '\t', w3_v)

from sklearn.metrics import r2_score, mean_absolute_error
predict = x1_data * w1_v + x2_data * w2_v + x3_data * w3_v + b_v
r2 = r2_score(y_data, predict)
mae = mean_absolute_error(y_data, predict)
print('r2' , r2 )
print('mae' , mae )
sess.close()