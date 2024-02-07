from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=13, stratify=y) 

# 모델구성
model = LinearSVC(C=100) #머신러닝에서 성능이 가장 안좋은 모델
#C가 크면 training 포인트를 정확히 구분(굴곡지다), C가 작으면 직선에 가깝다.

# 컴파일, 훈련
model.fit(x_train, y_train)

# 평가, 예측
from sklearn.metrics import accuracy_score
results = model.score(x_test, y_test)
print("model.score : ", results)#정확도:  1.0

x_predict = model.predict(x_test)
print(x_predict)

acc_score = accuracy_score(y_test, x_predict)
print("accuracy_score : ", acc_score)
'''
model.score :  1.0
[2 1 2 2 0 2 0 0 1 0 1 1 2 0 0 1 1 2 1 0 1 0 2 1 2 0 2 0 2 1]
accuracy_score :  1.0
'''