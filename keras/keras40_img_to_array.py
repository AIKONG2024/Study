import sys
import tensorflow as tf
print("tensorflow version: ", tf.__version__) #2.9.0
print("python version : ", sys.version) #3.9.18 

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import load_img #이미지를 가져옴
from tensorflow.keras.preprocessing.image import img_to_array #이미지를 수치화
# from keras.utils import img_to_array
# from keras.utils import load_img

path = "c:\_data\image\cat_and_dog\\train\Cat\\1.jpg"
img = load_img(path, 
            target_size = (150, 150)
)

print(img) #<PIL.Image.Image image mode=RGB size=150x150 at 0x26BF533AAC0>
print(type(img))
# plt.imshow(img)
# plt.show()

arr = img_to_array(img) #수치화
print(arr)
print(arr.shape)#(281, 300, 3) -> (150, 150, 3)
print(type(arr))#<class 'numpy.ndarray'>

#차원증가
img = np.expand_dims(arr, axis=0) #차원 늘림.
print(img.shape) #(1, 150, 150, 3)