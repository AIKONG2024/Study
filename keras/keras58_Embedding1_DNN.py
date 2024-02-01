from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
from sklearn.metrics import accuracy_score

#1. 데이터
docs = ['너무 재미있다', "참 최고에요", "참 잘만든 영화예요", 
        "추천하고 싶은 영화입니다.", "한 번 더 보고 싶어요" ,"글쎄", 
        "별로에요", "생각보다 지루해요", '연기가 어색해요', 
        '재미없어요', '너무 재미없다.', '참 재밋네요.', 
        '상헌이 바보', '반장 잘생겼다', '욱이 또 잔다'] 

label = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

'''
{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, 
'번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, 
'재미없어요': 21, '재미없다': 22, '재밋네요': 23, '상헌이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '욱이': 28, '또': 29, '잔다': 30}
'''

x = token.texts_to_sequences(docs)
pad_x = pad_sequences(x, padding = 'pre', 
                  maxlen=5,
                #   truncating='post' # maxlen보다 적으면 잘림. pre면 앞에가 잘림. post면 뒤에가 잘림.
                  )
print(pad_x)


'''
[[2, 3], [1, 4], [1, 5, 6], 
[7, 8, 9], [10, 11, 12, 13, 14], [15], 
[16], [17, 18], [19, 20], 
[21], [2, 22], [1, 23], 
[24, 25], [26, 27], [28, 29, 30]]
'''
print(pad_x.shape) #(15, 5)

#2.모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(128, input_shape = (5,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3.모델 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, label, epochs=1000, batch_size=1000)

#4.예측 결과
loss = model.evaluate(pad_x, label)
print('acc :', loss[1])

#===============================
# acc : 1.0