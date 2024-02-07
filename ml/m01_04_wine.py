from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구현
from sklearn.svm import LinearSVC
model = LinearSVC(C = 100)

#컴파일 훈련
model.fit(x_train, y_train)

#예측 평가
acc = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test)
print(y_predict)

acc_pred = accuracy_score(y_test, y_predict)
print("acc : ", acc)
print("eval_acc :", acc_pred)

'''
acc :  0.98
acc score : 0.98
'''