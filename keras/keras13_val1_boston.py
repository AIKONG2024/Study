
#데이터 가져오기
from sklearn.datasets import load_boston
datasets= load_boston()
x = datasets.data
y = datasets.target

#데이터 분석
print(x.shape)
print(y.shape)

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=20)

#데이터 구조 확인
print(x_train.shape)#(301, 13)
print(x_test.shape)#(152, 13)
print(y_train.shape)#(354,)
print(y_test.shape)#(152,)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim = 13))
model.add(Dense(10))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 10, batch_size= 10, validation_split=0.7, verbose=1)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('loss : ', loss)
print('result : ', y_predict)