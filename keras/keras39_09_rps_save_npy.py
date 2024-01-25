#150x150 rgb

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest'
)

train_path = 'C:/_data/image/rps/'

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(300,300),
    batch_size=1027,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

#이미지 저장
npy_path = 'C:/_data/_save_npy/rps/'
np.save(file= npy_path + 'keras39_07_save_x_train_rps.npy', arr= train_generator[0][0])
np.save(file= npy_path + 'keras39_07_save_y_train_rps.npy', arr= train_generator[0][1])