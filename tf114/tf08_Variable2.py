import tensorflow as tf
tf.compat.v1.set_random_seed(777)
tf.set_random_seed(777)

#[실습]
#07_2를 카피해서 만들기

#####1. Session() // sess.run(변수) ########
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #random_normal => random값, 정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 2. 모델구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)
train = optimizer.minimize(loss)
#3-2 훈련
with tf.compat.v1.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    #model.fit
    epochs = 101
    for step in range(epochs) : 
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict = {x:x_data, y:y_data})
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val) 
    x_pred_data = [6,7,8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    prediction = x_test * w_val + b_val
    prediction = sess.run(prediction, feed_dict = {x_test:x_pred_data})
    
    
#####2. Session() // 변수.eval(session=sess) ########
print("===========================================================")
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #random_normal => random값, 정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 2. 모델구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)
train = optimizer.minimize(loss)
#3-2 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #model.fit
    epochs = 101
    for step in range(epochs) : 
        _, loss_val= sess.run([train, loss], feed_dict = {x:x_data, y:y_data})
        w_val = w.eval(session= sess)
        b_val = b.eval(session= sess)
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val)
    x_pred_data = [6,7,8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    prediction = x_test * w_val + b_val
    prediction = sess.run(prediction, feed_dict = {x_test:x_pred_data})
    
    
#####3. InteractiveSession() // 변수.eval() ########
print("===========================================================")
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #random_normal => random값, 정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 2. 모델구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)
train = optimizer.minimize(loss)
#3-2 훈련
sess.run(tf.global_variables_initializer())

#model.fit
epochs = 101
for step in range(epochs) : 
    _, loss_val = sess.run([train, loss], feed_dict = {x:x_data, y:y_data})
    w_val = w.eval()
    b_val = b.eval()
    if step % 20 == 0:
        print(step, loss_val, w_val, b_val) 
x_pred_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
prediction = x_test * w_val + b_val
prediction = sess.run(prediction, feed_dict = {x_test:x_pred_data})