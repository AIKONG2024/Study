from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler,LabelEncoder
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

path = 'C:/_data/dacon/dechul/'
#데이터 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

unique, count = np.unique(train_csv['근로기간'], return_counts=True)
unique, count = np.unique(test_csv['근로기간'], return_counts=True)
train_le = LabelEncoder()
test_le = LabelEncoder()
train_csv['주택소유상태'] = train_le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = train_le.fit_transform(train_csv['대출목적'])
train_csv['근로기간'] = train_le.fit_transform(train_csv['근로기간'])
train_csv['대출등급'] = train_le.fit_transform(train_csv['대출등급'])


test_csv['주택소유상태'] = test_le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = test_le.fit_transform(test_csv['대출목적'])
test_csv['근로기간'] = test_le.fit_transform(test_csv['근로기간'])

#3. split 수치화 대상 int로 변경: 대출기간
train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(float)
test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(float)

x = train_csv.drop(["대출등급"], axis=1)
y = train_csv["대출등급"]

lbe = LabelEncoder()
y = lbe.fit_transform(y)

ohe = OneHotEncoder()
ohe_y = ohe.fit_transform(y.reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, ohe_y, random_state=42, train_size=0.7,stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape= [None,13])
y = tf.compat.v1.placeholder(tf.float32, shape= [None,7])

# w = tf.compat.v1.Variable(tf.random_normal([13,7]), name='weight')
# b = tf.compat.v1.Variable(tf.random_normal([1,7]), name='bias')

#2. 모델
layer1 = Layer(input_dim= 13, output_dim= 64, activation=tf.nn.relu).forward(x)
layer2 = Layer(input_dim= 64, output_dim= 32, activation=tf.nn.relu).forward(layer1)
layer3 = Layer(input_dim= 32, output_dim= 16, activation=tf.nn.sigmoid).forward(layer2)
output = Layer(input_dim= 16, output_dim= 7, activation=tf.nn.softmax).forward(layer3)

#3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(4000):
    _, loss_v= sess.run([train, loss], feed_dict = {x:x_train, y:y_train})
    print(loss_v, '\t')

predict =  output
predict = sess.run(predict, feed_dict={x:x_test})
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test,predict)

print('acc : ', acc)

sess.close()
#acc :  0.8218353006334591