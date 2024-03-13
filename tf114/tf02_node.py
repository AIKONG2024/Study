import tensorflow as tf

# 3 + 4 = ?
node1 = tf.constant(3.0, tf.float32) #데이터형 정의하지 않으면 자동으로 적힘
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)

print(node3)

sess = tf.Session()
print(sess.run(node3))

