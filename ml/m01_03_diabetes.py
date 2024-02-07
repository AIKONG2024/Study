from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.72, random_state=335688) 

# 모델 구성
from sklearn.svm import LinearSVC
model = LinearSVC(C=1)

# 컴파일 훈련
model.fit(x_train, y_train)

# 평가 예측
acc = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc_pred = accuracy_score(y_test, y_predict)
print("acc : ", acc)
print("eval_acc = ", acc_pred)

'''
acc :  0.008064516129032258
eval_acc =  0.008064516129032258
'''
