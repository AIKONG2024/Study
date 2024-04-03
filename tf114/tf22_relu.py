SEED = 11
import tensorflow as tf
tf.compat.v1.set_random_seed(SEED)
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MaxAbsScaler

class Layer:
    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.w = tf.compat.v1.Variable(tf.random.normal([input_dim, output_dim]), name='weight')
        self.b = tf.compat.v1.Variable(tf.zeros([output_dim]), name='bias')
    
    def forward(self, x):
        z = tf.matmul(x, self.w) + self.b
        if self.activation is not None:
            return self.activation(z)
        return z

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=SEED)
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(y, y.shape)

#2. 모델
x_p = tf.compat.v1.placeholder(tf.float32, shape=[None,30])
y_p = tf.compat.v1.placeholder(tf.float32, shape=[None,])
keep_prob = tf.compat.v1.placeholder(tf.float32)

layer1 = Layer(input_dim= 30, output_dim= 64, activation=tf.nn.relu).forward(x_p)
layer2 = Layer(input_dim= 64, output_dim= 32, activation=tf.nn.relu).forward(layer1)
layer2 = tf.compat.v1.nn.dropout(layer2, keep_prob=keep_prob)
output = Layer(input_dim= 32, output_dim= 1, activation=tf.nn.sigmoid).forward(layer2)

#3-1. 컴파일 
loss = -tf.reduce_mean(y_p*tf.log(output + 1e-7) + (1-y_p)*tf.log(1-output + 1e-7)) #binary crossentropy

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
for step in range(10000):
    _, loss_v = sess.run([train, loss], feed_dict = {x_p:x_train, y_p:y_train, keep_prob:0.5})
    if step % 100 == 0:
        print(step, loss_v)

#4. 평가 및 예측
predict = sess.run(output, feed_dict={x_p: x_test, keep_prob:1.0})
predict = np.round(predict)
acc = accuracy_score(y_test, predict)
print('acc' , acc ) 
sess.close()

#acc 0.5555555555555556