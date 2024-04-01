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

######################요기부터 증폭 #######################
datagen = ImageDataGenerator(
    # horizontal_flip= True,
    # vertical_flip=True,
    # width_shift_range=0.2,
    # height_shift_range=0.5,
    # zoom_range=0.5,
    # rotation_range=180,
    shear_range=10, #기울기,
    rescale=1./255
)

it = datagen.flow(img,
                  batch_size=1, 
                  )
fig, ax = plt.subplots(nrows=3, ncols=5, figsize = (20,20))

for i in range(15):
    batch = it.next()
    image = batch[0]
    ax[i//5, i%5].imshow(image) #1/5, 1%5의 정수값
    ax[i//5, i%5].axis('off') #0/5, 0%5
plt.show()