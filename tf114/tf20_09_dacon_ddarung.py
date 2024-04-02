#7load diabetes
#8california
#9dacon 따릉
#10kaggle bike

import tensorflow as tf
import pandas as pd
tf.compat.v1.set_random_seed(777)
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
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
# 보간법 - 결측치 처리
from sklearn.impute import KNNImputer
#KNN
imputer = KNNImputer(weights='distance')
train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)

x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
y = train_csv['count'] 

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=777)
x_p = tf.compat.v1.placeholder(tf.float32, shape=[None,9])
y_p = tf.compat.v1.placeholder(tf.float32, shape=[None,])
print(y_train.shape) #(1167,)
# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9,1], name='weights'))
# b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias'))

#2. 모델
layer1 = Layer(input_dim= 9, output_dim= 64, activation=tf.nn.relu).forward(x_p)
layer2 = Layer(input_dim= 64, output_dim= 32, activation=tf.nn.relu).forward(layer1)
layer3 = Layer(input_dim= 32, output_dim= 16, activation=tf.nn.relu).forward(layer2)
output = Layer(input_dim= 16, output_dim= 1).forward(layer3)

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(output - y_p)) #mse

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())
#3-2. 훈련
for step in range(3000):
    _, loss_v = sess.run([train, loss], feed_dict={x_p: x_train, y_p: y_train})
    print(step, '\t', loss_v, '\t')

# 예측
predict = sess.run(output, feed_dict={x_p: x_test})
# 평가
r2 = r2_score(y_test, predict)
mae = mean_absolute_error(y_test, predict)
print('r2:', r2)
print('mae:', mae)

#저장
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"submission_{save_time}.csv"
submission_csv.to_csv(file_path, index=False)

'''
r2: 0.01501594134018669
mae: 63.59610139507137
'''