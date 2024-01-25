#300x300으로 고정
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest'
)

train_path = 'C:/_data/image/horse_human/'

train_c_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(300,300),
    batch_size=1027,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)


#이미지 저장
npy_path = 'C:/_data/_save_npy/horse_human/'
#categorical
np.save(file= npy_path + 'keras39_07_save_x_train_horse_b_human.npy', arr= train_c_generator[0][0])
np.save(file= npy_path + 'keras39_07_save_y_train_horse_b_human.npy', arr= train_c_generator[0][1])