from sklearn.metrics import r2_score
import time
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape, y.shape)  # (442, 10) (442,)

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

# R2 0.62 이상.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.72, random_state=335688)  # 335688

# 모델 구성

model = Sequential()
model.add(Dense(64, input_dim=10))
model.add(Dense(32))
model.add(Dense(1))

#Early Stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1, restore_best_weights=True)

# 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'acc'])
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=20, validation_split=0.7, callbacks=[es])
end_time = time.time()

# 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("r2 = ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

#로스값, val_loss
hist_loss = hist.history['loss']
hist_val_loss = hist.history['val_loss']

#시각화
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.figure(figsize=(6,6))
plt.plot(hist_loss, c = 'red', label = 'loss', marker = '.')
plt.plot(hist_val_loss, c = 'blue', label = 'val_loss', marker = '.')
plt.legend(loc = 'upper right')
plt.xlabel = 'epoch'
plt.ylabel = 'loss'
plt.title = '디아벳 차트'
plt.grid()
plt.show()

#restore_best_weights = True
