from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return "XGBClassifier()"

# 1. 데이터
# x, y = datasets = load_iris(return_X_y=True)
datasets = load_iris()
x = datasets.data
y = datasets.target

#넘파이로 삭제
# x = np.delete(x, 0, axis=1)
#판다스로 삭제
x = pd.DataFrame(data = x, columns = datasets.feature_names).drop(datasets.feature_names[0], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=13, stratify=y) 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), CustomXGBClassifier(random_state=random_state)]
for model in models :

    # 컴파일, 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    from sklearn.metrics import accuracy_score
    results = model.score(x_test, y_test)
    print(f"[{type(model).__name__}] model.score : ", results)

    x_predict = model.predict(x_test)
    acc_score = accuracy_score(y_test, x_predict)
    print(f"[{type(model).__name__}] model accuracy_score : ", acc_score)

    print(type(model).__name__ ,":", model.feature_importances_)
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
[CustomXGBClassifier] model.score :  1.0
[CustomXGBClassifier] model accuracy_score :  1.0
CustomXGBClassifier : [0.01490525 0.02423168 0.7842199  0.17664316]

======================================
#하위 25% 컬럼 제거후
[DecisionTreeClassifier] model.score :  1.0
[DecisionTreeClassifier] model accuracy_score :  1.0
DecisionTreeClassifier : [0.01666667 0.57742557 0.40590776]
[RandomForestClassifier] model.score :  1.0
[RandomForestClassifier] model accuracy_score :  1.0
RandomForestClassifier : [0.14488227 0.45185802 0.40325971]
[GradientBoostingClassifier] model.score :  1.0
[GradientBoostingClassifier] model accuracy_score :  1.0
GradientBoostingClassifier : [0.02169268 0.57532013 0.40298719]
[CustomXGBClassifier] model.score :  1.0
[CustomXGBClassifier] model accuracy_score :  1.0
CustomXGBClassifier : [0.03740263 0.7588328  0.20376457]
'''
    
