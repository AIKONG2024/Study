#Train Test 를 분리해서 해보기
#불러오는데 걸리는 시간.

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os


image_path = 'C:\_data\image\cat_and_dog/test/rand/'
path = 'C:\_data\kaggle\cat_and_dog/'
np_path = '../_data/_save_npy/'
x_train = np.load(np_path + 'keras39_3_x_train.npy')
y_train = np.load(np_path + 'keras39_3_y_train.npy')
x_test = np.load(np_path + 'keras39_3_x_test.npy')


# print(x_train.shape)
# print(x_test.shape)

# x_train = x_train/255.
# x_test = x_test/255.

#모델구성
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM
model = Sequential()

model.add(LSTM(32, input_shape=(100*300,1), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', patience=300, restore_best_weights=True)

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs= 1000, batch_size= 5, validation_split= 0.2, callbacks=[es])

#평가 예측
predict = np.round(model.predict(x_test)).flatten()

print(predict)
print(len(predict))
file = os.listdir(image_path)
for i in range(len(file)):
    file[i] = file[i].replace('.jpg', '')
#제출
data = pd.DataFrame({'Id':file, 'Target': predict})

import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"sampleSubmission{save_time}.csv"
data.to_csv(file_path, index=False)

'''
loss :  0.46612799167633057
acc :  0.7822499871253967

loss :  0.4787946939468384
acc :  0.7768844366073608

=============RNN 적용

'''