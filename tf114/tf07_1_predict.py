import tensorflow as tf
tf.set_random_seed(777)

# 1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

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
print(train)
#3-2 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #model.fit
    epochs = 2000
    for step in range(epochs) : 
        # sess.run(train, feed_dict={x:[1,2,3,4,5], y : [3,5,7,9,11]})
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict = {x:x_data, y:y_data}) #train이 주연산, 주연산에 들어가는 건 x, y
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val) #verbose와 model.weight에서 확인했던 애들.

    # sess.close() #세션은 항상 닫아줘야함
    
    
    #################################실습 ###################################
    x_pred_data = [6,7,8]
    #예측값을 뽑기
    #########################################
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    #1. 파이썬 방식
    # prediction = x_pred_data * w_val + b_val
    
    #2. placeholder 방식 - 이렇게 넣는게 더 깔끔함.
    prediction = x_test * w_val + b_val
    prediction = sess.run(prediction, feed_dict = {x_test:x_pred_data})
    # print('2[6, 7, 8]의 예측 :', prediction)
    #2[6, 7, 8]의 예측 : [13.000149 15.00021  17.000273]
    
    #3, 모델이 커지면 hypothesis를 사용하면 문제가 생길 수 있음.
    prediction = sess.run(hypothesis, feed_dict = {x:x_pred_data})
    print('3[6, 7, 8]의 예측 :', prediction)
    #3[6, 7, 8]의 예측 : [13.000149 15.00021  17.000273]