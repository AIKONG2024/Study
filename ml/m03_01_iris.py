from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
x, y = datasets = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=13, stratify=y) 

# 모델구성
# model = LinearSVC(C=100) 
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

models = [LinearSVC(), Perceptron(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models :
    # 컴파일, 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    from sklearn.metrics import accuracy_score
    results = model.score(x_test, y_test)
    print(f"[{type(model).__name__}] model.score : ", results)#정확도:  1.0

    x_predict = model.predict(x_test)
    # print(x_predict)
    acc_score = accuracy_score(y_test, x_predict)
    print(f"[{type(model).__name__}] model accuracy_score : ", acc_score)
    '''
    [LinearSVC] model.score :  1.0
    [LinearSVC] model accuracy_score :  1.0
    [Perceptron] model.score :  0.6666666666666666
    [Perceptron] model accuracy_score :  0.6666666666666666
    [LogisticRegression] model.score :  1.0
    [LogisticRegression] model accuracy_score :  1.0
    [KNeighborsClassifier] model.score :  0.9666666666666667
    [KNeighborsClassifier] model accuracy_score :  0.9666666666666667
    [DecisionTreeClassifier] model.score :  0.9666666666666667
    [DecisionTreeClassifier] model accuracy_score :  0.9666666666666667
    [RandomForestClassifier] model.score :  1.0
    [RandomForestClassifier] model accuracy_score :  1.0
    '''