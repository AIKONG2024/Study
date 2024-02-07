from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.72, random_state=335688) 

# 모델 구성
models = [LinearSVC(), Perceptron(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models :
    # 컴파일 훈련
    model.fit(x_train, y_train)

    # 평가 예측
    acc = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    acc_pred = accuracy_score(y_test, y_predict)
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)

'''
[LinearSVC] model acc :  0.008064516129032258
[LinearSVC] eval_acc :  0.008064516129032258
[Perceptron] model acc :  0.0
[Perceptron] eval_acc :  0.0
[LogisticRegression] model acc :  0.008064516129032258
[LogisticRegression] eval_acc :  0.008064516129032258
[KNeighborsClassifier] model acc :  0.016129032258064516
[KNeighborsClassifier] eval_acc :  0.016129032258064516
[DecisionTreeClassifier] model acc :  0.0
[DecisionTreeClassifier] eval_acc :  0.0
[RandomForestClassifier] model acc :  0.0
[RandomForestClassifier] eval_acc :  0.0
'''
