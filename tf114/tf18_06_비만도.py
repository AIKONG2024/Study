from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler,LabelEncoder

# get data
path = "C:/_data/kaggle/obesity/"
train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + "test.csv")

# train_csv = colume_preprocessing(train_csv)

train_csv['BMI'] =  train_csv['Weight'] / (train_csv['Height'] ** 2)
test_csv['BMI'] =  test_csv['Weight'] / (test_csv['Height'] ** 2)

lbe = LabelEncoder()
cat_features = train_csv.select_dtypes(include='object').columns.values
for feature in cat_features :
    train_csv[feature] = lbe.fit_transform(train_csv[feature])
    if feature == "CALC" and "Always" not in lbe.classes_ :
        lbe.classes_ = np.append(lbe.classes_, "Always")
    if feature == "NObeyesdad":
        continue
    test_csv[feature] = lbe.transform(test_csv[feature]) 
                
x, y = train_csv.drop(["NObeyesdad"], axis=1), train_csv.NObeyesdad

ohe = OneHotEncoder()
ohe_y = ohe.fit_transform(y.values.reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, ohe_y, random_state=777, train_size=0.8,stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape= [None,18])
y = tf.compat.v1.placeholder(tf.float32, shape= [None,7])

w = tf.compat.v1.Variable(tf.random_normal([18,7]), name='weight')
b = tf.compat.v1.Variable(tf.random_normal([1,7]), name='bias')

#2. 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

#3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.2)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
for step in range(100):
    _, loss_v, w_v, b_v = sess.run([train, loss, w, b], feed_dict = {x:x_train, y:y_train})
    print(loss_v, '\t', w_v, '\t', b_v)

predict =  tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)
predict = sess.run(predict, feed_dict={x:x_test})
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test,predict)

print('acc : ', acc)

sess.close()
#acc :  0.7507225433526011