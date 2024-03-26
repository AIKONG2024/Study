import tensorflow as tf
tf.set_random_seed(777)

# 1. 데이터
x = [1,2,3,4,5]
y = [3,5,7,9,11]

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype= tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #random_normal => random값, 정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w))

# 2. 모델구성
# y =wx+b
# y = w * x + b ==> y = x * w + b
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
# model.complie(loss = 'mse', optimizer = 'sgd')

#3-2 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #model.fit
    epochs = 2000
    for step in range(epochs) : 
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b)) #verbose와 model.weight에서 확인했던 애들.

    # sess.close() #세션은 항상 닫아줘야함
