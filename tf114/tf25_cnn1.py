import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, [None, 28,28,1]) #input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 10]) 

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64]) #kernel 사이즈, 채널, output
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') #stride1 에 4차원이므로 1,1,1,1  stride 2,2 ==> 1,2,2,1
#model.add(Conv2d(64, kernel_size=(2,2), stride = (1,1), input_shape=(28,28,1)))

# print(w1) #summary
# print (L1)

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3,3,64,32]) #kernel_size: 3,3 // L2 채널 : L1의 output 값 64, output : 32 
L2 = tf.nn.conv2d(L1, w2, strides=[1,2,2,1], padding='SAME') #strides : 2

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[2,2,32,16])
L2 = tf.nn.conv2d(L2, w3, strides=[1,2,2,1], padding='SAME')

print(w3)
print(L2)
