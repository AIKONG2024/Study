import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,          # 각각의 이미지 사이즈를 맞춰줌
    horizontal_flip=True,    # 수평 뒤집기
    vertical_flip=True,      # 수직 뒤집기
    width_shift_range=0.1,   # 평행이동
    height_shift_range=0.1,  # 평행이동
    rotation_range=5,        # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,          # 축소 또는 확대
    shear_range=0.7,         # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest',     # 이동해서 빈 값을 0이 아닌 최종값과 유사한 근사값으로 정해줌.
)

test_datagen = ImageDataGenerator(
    rescale=1./255,         # 각각의 이미지 사이즈를 맞춰줌
)

path_train = 'c:/_data/image/brain/train/' #하단의 폴더가 라벨이됨. ad, normal: 라벨에 맞는 사진들을 모아 놓아야함.
path_test = 'c:/_data/image/brain/test/' 

xy_train =  train_datagen.flow_from_directory(
    path_train,
    target_size= (200,200),
    batch_size=160,          #160개를 10인 것으로 잘라 10.....10 16개가 생성됨
    class_mode='binary',
    shuffle=True,
)#Found 160 images belonging to 2 classes.


xy_test =  test_datagen.flow_from_directory(
    path_test,
    target_size= (200,200),
    batch_size=10,
    class_mode='binary',
)#Found 120 images belonging to 2 classes.
print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x000001C4C4BA3520>
print(xy_test)
#<keras.preprocessing.image.DirectoryIterator object at 0x000001825415D2E0>
print(xy_train.next()) #첫번째값
print(xy_train[0])
# print(xy_train[16]) #에러 :: 전체데이터/batch_size = 160/10 = 16개인데 17번째 값을 빼라고 해서 에러
print(xy_train[0][0]) # 첫번째 배치의 X
print(xy_train[0][1]) # 첫번쨰 배치의 y
print(xy_train[1][0].shape) #2번째 배치의 입력값 형태(10, 200, 200, 3)

print(type(xy_train[0]))#Tuple x가 튜플 첫번째, y가 튜블 두번쨰.
print(type(xy_train[0][0]))#<class 'numpy.ndarray'>  : X
print(type(xy_train[0][1]))#<class 'numpy.ndarray'>  ; Y
print(len(xy_train[0][0]))
print(len(xy_train[0][1]))
