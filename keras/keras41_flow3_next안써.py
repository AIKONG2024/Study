from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./ 255, 
    horizontal_flip=True, 
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode="nearest",
)

augumet_size = 100
#augumet : 증식
print(np.tile(x_train[0].reshape(28 * 28), augumet_size).shape)
print(np.zeros(augumet_size).shape)
x_data = train_datagen.flow( # 수치를 받음.
    np.tile(x_train[0].reshape(28 * 28), augumet_size).reshape(-1, 28, 28, 1),  
    np.zeros(augumet_size), #y값은 구색만 갖추겠음.
    batch_size=augumet_size,
    shuffle=False,
) #.next()
print(x_data[0][0].shape)
print(x_data[0][1].shape)
print(np.unique(x_data[0][1], return_counts=True)) #(array([0.]), array([100], dtype=int64))
print(x_data[0][0][0].shape)


plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()

