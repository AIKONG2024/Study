#가상환경 tf215_gpu
import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus :
    print("gpu o")
else:
    print('gpu x')