import tensorflow as tf
tf.set_random_seed(777)
import matplotlib.pyplot as plt

# 1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #random_normal => random값, 정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w))

# 2. 모델구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)
train = optimizer.minimize(loss)

#3-2 훈련
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 101
    for step in range(epochs) : 
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict = {x:x_data, y:y_data}) #train이 주연산, 주연산에 들어가는 건 x, y
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val) #verbose와 model.weight에서 확인했던 애들.
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
    #################################실습 ###################################
    x_pred_data = [6,7,8]
    #########################################
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    prediction = x_test * w_val + b_val
    prediction = sess.run(prediction, feed_dict = {x_test:x_pred_data})
    
# print(loss_val_list)
# print(w_val_list)
plt.figure(figsize=(9, 9))  # 전체 그림 크기 설정
plt.subplot(2,2,1)
plt.plot(loss_val_list, c = 'red')
plt.xlabel('epochs')
plt.ylabel('losses')

plt.subplot(2,2,2)
plt.plot(w_val_list, c = 'green' )
plt.xlabel('epochs')
plt.ylabel('weights')

plt.subplot(2,2,3)
plt.scatter(w_val_list,loss_val_list)
plt.xlabel('weights')
plt.ylabel('losses')


plt.show()