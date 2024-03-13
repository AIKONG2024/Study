import tensorflow as tf
print(tf.__version__)

print("tensorflow hello world")

hello = tf.constant('hello world')
print(hello)

#그냥 출력하지 않고 session . run 을 꼭 거쳐야함
sess = tf.Session()
print(sess.run(hello))