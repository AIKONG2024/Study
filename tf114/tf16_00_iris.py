SEED = 5652
import tensorflow as tf
tf.compat.v1.set_random_seed(SEED)
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris
#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x = x[ y != 2]
y = y[ y != 2]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=SEED)

# print(y, y.shape)

x_p = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y_p = tf.compat.v1.placeholder(tf.float32, shape=[None,])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,1]),name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')

#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x_p, w) + b) #activation : sigmoid --> hypothesis를 sigmoid함수로 감싸주면 됨. 

#3-1. 컴파일 
loss = -tf.reduce_mean(y_p*tf.log(hypothesis) + (1-y_p)*tf.log(1-hypothesis)) #binary crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000005)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
for step in range(500):
    _, loss_v, w_v, b_v = sess.run([train, loss, w, b], feed_dict = {x_p:x_train, y_p:y_train})
    print(step, '\t', loss_v, '\t')

#4. 평가 및 예측
predict = tf.compat.v1.sigmoid(tf.matmul(x_p, w) + b)
predict = sess.run(predict, feed_dict={x_p: x_test})
predict = np.around(predict)
acc = accuracy_score(y_test, predict)
print('acc' , acc ) 
sess.close()

#acc 0.55