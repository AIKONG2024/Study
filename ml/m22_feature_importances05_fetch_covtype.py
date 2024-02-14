from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

x,y = fetch_covtype(return_X_y=True)
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
y = lbe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234, stratify=y)
print(x_test.shape)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]
for model in models:

    #컴파일, 훈련
    model.fit(x_train, y_train)

    #평가 예측
    acc = model.score(x_test, y_test)
    y_predict = lbe.inverse_transform(model.predict(x_test)) 

    acc_pred = accuracy_score(y_test, y_predict)
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)
    print(type(model).__name__ ,":", model.feature_importances_)

'''
[LogisticRegression] model acc :  0.7229495593904902
[LogisticRegression] eval_acc :  0.7229495593904902
[KNeighborsClassifier] model acc :  0.9258249954103176
[KNeighborsClassifier] eval_acc :  0.9258249954103176
[DecisionTreeClassifier] model acc :  0.9352281072149807
[DecisionTreeClassifier] eval_acc :  0.9352281072149807
[RandomForestClassifier] model acc :  0.9532081879933909
[RandomForestClassifier] eval_acc :  0.9532081879933909
'''