#7load diabetes
#8california
#9dacon 따릉
#10kaggle bike

import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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

#1.데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)#(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=777)
x_p = tf.compat.v1.placeholder(tf.float32, shape=[None,10])
y_p = tf.compat.v1.placeholder(tf.float32, shape=[None,])
print(y_train.shape) #(353,)
# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], name='weights'))
# b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias'))

#2. 모델
layer1 = Layer(input_dim= 10, output_dim= 64, activation=tf.nn.relu).forward(x_p)
layer2 = Layer(input_dim= 64, output_dim= 32, activation=tf.nn.relu).forward(layer1)
layer3 = Layer(input_dim= 32, output_dim= 16, activation=tf.nn.relu).forward(layer2)
output = Layer(input_dim= 16, output_dim= 1).forward(layer3)

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(output - y_p)) #mse

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())
#3-2. 훈련
for step in range(5000):
    _, loss_v = sess.run([train, loss], feed_dict={x_p: x_train, y_p: y_train})
    print(step, '\t', loss_v, '\t')

# 예측
predict_val = sess.run(output, feed_dict={x_p: x_test})
# 평가
r2 = r2_score(y_test, predict_val)
mae = mean_absolute_error(y_test, predict_val)
print('r2:', r2)
print('mae:', mae)
'''
r2: -0.013393987788295814
mae: 62.021712742494735
'''