import tensorflow as tf
print(tf.__version__) #1.14.0
print(tf.executing_eagerly()) #false #즉시실행모드

# 즉시실행모드 -> 텐서 1의 그래프 형태의 구성없이 파이썬 문법으로 실행
# tf.enable_eager_execution() #True
tf.compat.v1.disable_eager_execution() #즉시실행모드 끔 // 텐서플로 1.0 문법// 디폴트
# tf.compat.v1.enable_eager_execution() #즉시실행모드 킴 // 텐서플로 2.0 사용가능

print(tf.executing_eagerly()) #True

hello = tf.constant('Hello World')

sess = tf.compat.v1.Session()
print(sess.run(hello))

# 가상환경   즉시실행모드     사용가능
# 1.14.0    disable(디폴트)   가능
# 1.14.0    enable            에러 
# 2.9.0     disable           가능 
# 2.9.0     enable(디폴트)    에러

#즉, tensor 2에서 tensor1 코드를 사용하고 싶다면 disable를 설정