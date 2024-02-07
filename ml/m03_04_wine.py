from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

x,y = load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구현
models = [LinearSVC(), Perceptron(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]

for model in models :
    #컴파일 훈련
    model.fit(x_train, y_train)

    #예측 평가
    acc = model.score(x_test, y_test)
    y_predict = model.predict(x_test)

    acc_pred = accuracy_score(y_test, y_predict)
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)

'''
[LinearSVC] model acc :  0.98
[LinearSVC] eval_acc :  0.98
[Perceptron] model acc :  0.98
[Perceptron] eval_acc :  0.98
[LogisticRegression] model acc :  1.0
[LogisticRegression] eval_acc :  1.0
[KNeighborsClassifier] model acc :  0.92
[KNeighborsClassifier] eval_acc :  0.92
[DecisionTreeClassifier] model acc :  0.88
[DecisionTreeClassifier] eval_acc :  0.88
[RandomForestClassifier] model acc :  0.98
[RandomForestClassifier] eval_acc :  0.98
'''