import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())

tf.compat.v1.disable_eager_execution() # 설정시에 2버전 1버전 모두 가능할 것 GPU에 연결

node1 = tf.constant(30.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4}))
print(sess.run(add_node, feed_dict={a:30, b:4.5}))

add_and_triple = add_node * 3
print(sess.run(add_and_triple, feed_dict={a:3, b:4}))