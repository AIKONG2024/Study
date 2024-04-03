SEED = 7878
import tensorflow as tf
tf.compat.v1.set_random_seed(SEED)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
class Layer:
    def __init__(self, input_dim, output_dim, activation=None, name=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        with tf.compat.v1.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.w = tf.compat.v1.get_variable(f'{name}_weight', [input_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.compat.v1.get_variable(f'{name}_bias', [output_dim], initializer=tf.zeros_initializer())
    
    
    def forward(self, x):
        z = tf.matmul(x, self.w) + self.b
        if self.activation is not None:
            return self.activation(z)
        return z
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

x_train = x_train.reshape(60000, 28*28).astype('float32')/ 255
x_test = x_test.reshape(10000, 28*28).astype('float32')/ 255

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None,784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

layer1 = Layer(input_dim= 784, output_dim= 64, activation=tf.nn.relu, name='layer1').forward(x)
layer2 = Layer(input_dim= 64, output_dim= 32, activation=tf.nn.relu, name='layer2').forward(layer1)
layer3 = Layer(input_dim= 32, output_dim= 32, activation=tf.nn.relu, name='layer3').forward(layer2)
layer4 = Layer(input_dim= 32, output_dim= 16, activation=tf.nn.sigmoid, name='layer4').forward(layer3)
layer5 = Layer(input_dim= 16, output_dim= 8, activation=tf.nn.relu, name='layer5').forward(layer4)
output = Layer(input_dim= 8, output_dim= 10, activation=tf.nn.softmax, name='output').forward(layer5)

#3-1 컴파일
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), axis=1))
loss = tf.compat.v1.losses.softmax_cross_entropy(y, output)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.003)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(1000):
    _, loss_v= sess.run([train, loss], feed_dict = {x:x_train, y:y_train})
    print(step, loss_v)

predict =  output
predict = sess.run(predict, feed_dict={x:x_test})
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test,predict)

print('acc : ', acc)

sess.close()
#acc :  0.9414