from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
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

path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

#데이터 확인
print(train_csv.shape)#(5497, 13)
print(test_csv.shape)#(1000, 12)
print(submission_csv.shape)#(1000, 2) "species"

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']

#결측치 확인
print(x.isna().sum())
print(y.isna().sum())

#분류 클래스 확인
print(pd.value_counts(y)) #(array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
print(x.shape)#(5497, 12) 입력값: 12 출력값: 7
print(y.shape)#(5497,)

#OneHotEncoder
# scikit learn 방식
from sklearn.preprocessing import OneHotEncoder
y = y.values.reshape(-1,1) 
one_hot_y = OneHotEncoder(sparse=False).fit_transform(y)

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, one_hot_y, train_size=0.7, random_state=42, stratify=one_hot_y)
print(np.unique(y_test, return_counts=True))
#(array([False,  True]), array([9900, 1650], dtype=int64))

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape= [None,12])
y = tf.compat.v1.placeholder(tf.float32, shape= [None,7])

#2. 모델
layer1 = Layer(input_dim= 12, output_dim= 8, activation=tf.nn.relu).forward(x)
layer2 = Layer(input_dim= 8, output_dim= 4, activation=None).forward(layer1)
layer3 = Layer(input_dim= 4, output_dim= 2, activation=tf.nn.sigmoid).forward(layer2)
output = Layer(input_dim= 2, output_dim= 7, activation=tf.nn.softmax).forward(layer3)

#3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(100):
    _, loss_v= sess.run([train, loss], feed_dict = {x:x_train, y:y_train})
    print(loss_v)

predict =  output
predict = sess.run(predict, feed_dict={x:x_test})
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test,predict)

print('acc : ', acc)

sess.close()
#acc :  0.4393939393939394