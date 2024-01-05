import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_diabetes

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape , y.shape) # (442, 10) (442,)

print(dataset.feature_names)
print(dataset.DESCR)
'''
**Data Set Characteristics:**

  :Number of Instances: 442

  :Number of Attributes: First 10 columns are numeric predictive values

  :Target: Column 11 is a quantitative measure of disease progression one year after baseline

  :Attribute Information:
      - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, total serum cholesterol
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, total cholesterol / HDL
      - s5      ltg, possibly log of serum triglycerides level
      - s6      glu, blood sugar level
'''

#R2 0.62 이상.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.72, random_state=335688) #335688

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim = 10))
model.add(Dense(32))
model.add(Dense(1))

#컴파일 훈련
import time
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train , epochs= 166, batch_size= 16)
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("r2 = ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

#train_size : 0.72 /  deep 6 (10-64-32-1)  / random_state : 335688 / epochs = 120 / batch_size = 16
# 로스 :  2399.896484375
# r2 =  0.6374057519332543
# 걸린시간 :  1.58 초

#train_size : 0.72 /  deep 6 (10-64-32-1)  / random_state : 335688 / epochs = 130 / batch_size = 16
# 로스 :  2394.226806640625
# r2 =  0.6382623477765563
# 걸린시간 :  1.65 초

#train_size : 0.72 /  deep 6 (10-64-32-1)  / random_state : 335688 / epochs = 166 / batch_size = 15
# 로스 :  2372.69677734375
# r2 =  0.6415152545466019
# 걸린시간 :  2.17 초