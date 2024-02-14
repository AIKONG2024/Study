from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
import numpy as np
from xgboost import XGBClassifier

# 1. 데이터
x, y = datasets = load_iris(return_X_y=True)
# print(np.unique(y, return_counts=True))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=13, stratify=y) 


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]
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

    print(type(model).__name__ ,":", model.feature_importances_) #중요도/ 낮다고 지우는건 위험함. 성능이 확 떨어짐
    '''
    [DecisionTreeClassifier] model.score :  0.9666666666666667
    [DecisionTreeClassifier] model accuracy_score :  0.9666666666666667
    DecisionTreeClassifier : [0.         0.05       0.88920455 0.06079545]
    [RandomForestClassifier] model.score :  1.0
    [RandomForestClassifier] model accuracy_score :  1.0
    RandomForestClassifier : [0.11419404 0.0211338  0.44587773 0.41879444]
    [GradientBoostingClassifier] model.score :  1.0
    [GradientBoostingClassifier] model accuracy_score :  1.0
    GradientBoostingClassifier : [0.00416537 0.02086211 0.6306077  0.34436482]
    [XGBClassifier] model.score :  1.0
    [XGBClassifier] model accuracy_score :  1.0
    XGBClassifier : [0.01490525 0.02423168 0.7842199  0.17664316]
    '''