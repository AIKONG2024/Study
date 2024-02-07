from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import time

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234, stratify=y)
print(x_test.shape)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구성
from sklearn.svm import LinearSVC
model = LinearSVC(C=100)

#컴파일, 훈련
start_time = time.time()
history = model.fit(x_train, y_train)
end_time = time.time()

#평가 예측
acc = model.score(x_test, y_test)
y_predict = model.predict(x_test)

acc_score = accuracy_score(y_test, y_predict)
print("acc : ", acc)
print("eval acc  : ", acc_score)
#걸린시간 측정 CPU GPU 비교
print("걸린시간 : ", round(end_time - start_time, 2), "초")

'''
acc :  0.6333130622360933
eval acc  :  0.6333130622360933
걸린시간 :  223.45 초
'''