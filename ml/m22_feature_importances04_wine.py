from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

x,y = load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]

for model in models :
    #컴파일 훈련
    model.fit(x_train, y_train)

    #예측 평가
    acc = model.score(x_test, y_test)
    y_predict = model.predict(x_test)

    acc_pred = accuracy_score(y_test, y_predict)
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)
    print(type(model).__name__ ,":", model.feature_importances_)

'''
[DecisionTreeClassifier] model acc :  0.88
[DecisionTreeClassifier] eval_acc :  0.88
DecisionTreeClassifier : [0.         0.         0.         0.         0.         0.
 0.41807051 0.         0.         0.39999374 0.         0.02317786
 0.15875789]
[RandomForestClassifier] model acc :  0.98
[RandomForestClassifier] eval_acc :  0.98
RandomForestClassifier : [0.12120274 0.04812494 0.01324031 0.02515674 0.03911373 0.05360255
 0.20101047 0.0129357  0.02501357 0.16493558 0.08091217 0.06910194
 0.14564955]
[GradientBoostingClassifier] model acc :  0.9
[GradientBoostingClassifier] eval_acc :  0.9
GradientBoostingClassifier : [1.03551899e-02 3.72715794e-02 4.81148468e-03 1.81408683e-03
 1.96796288e-03 3.71682289e-03 3.19469538e-01 4.54461591e-04
 4.45998506e-06 3.01426388e-01 4.84527660e-03 4.50825372e-02
 2.68780212e-01]
[XGBClassifier] model acc :  0.94
[XGBClassifier] eval_acc :  0.94
XGBClassifier : [0.02886936 0.0670943  0.01873133 0.01474714 0.01018555 0.02379604
 0.25580558 0.         0.01095851 0.2815821  0.02277394 0.03877123
 0.226685  ]
'''