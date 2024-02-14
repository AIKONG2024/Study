#https://dacon.io/competitions/open/236070/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

path = "C:/_data/dacon/iris/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

#데이터 확인
x = train_csv.drop(columns='species')
y = train_csv['species']

# 판다스로 삭제
x = x.drop(
    [
        "sepal length (cm)",
    ],
    axis=1,
)
test_csv = test_csv.drop(
    [
        "sepal length (cm)",
    ],
    axis=1,
)


#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=200, stratify=y)


random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]
for model in models:
    #컴파일 , 훈련
    model.fit(x_train, y_train)

    #평가, 예측
    acc = model.score(x_test, y_test)
    y_predict = model.predict(x_test)

    acc_pred = accuracy_score(y_test, y_predict) 
    submission = model.predict(test_csv)
    submission_csv['species'] = submission
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)
    # print(type(model).__name__ ,":", model.feature_importances_)
    
    # 25%이하 컬럼
    # feature_importance_set = pd.DataFrame({'feature': x.columns, 'importance':model.feature_importances_})
    # feature_importance_set.sort_values('importance', inplace=True)
    # delete_0_25_features = feature_importance_set['feature'][:int(len(feature_importance_set.values) * 0.25)]
    # delete_0_25_importance = feature_importance_set['importance'][:int(len(feature_importance_set.values) * 0.25)]
    # print(f'''
    # 제거 할 컬럼명 : 
    # {delete_0_25_features}  
    # 제거 할 feature_importances_ : 
    # {delete_0_25_importance}    
    # ''')
'''
[XGBClassifier] model acc :  0.9722222222222222
[XGBClassifier] eval_acc :  0.9722222222222222
=======================
제거후
[XGBClassifier] model acc :  0.9722222222222222
[XGBClassifier] eval_acc :  0.9722222222222222
'''
