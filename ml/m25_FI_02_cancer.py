import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
# x,y = load_breast_cancer(return_X_y=True)
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# 넘파이로 삭제
# x = np.delete(x, 0, axis=1)
# 판다스로 삭제
x = pd.DataFrame(data=x, columns=datasets.feature_names)
x = x.drop(
    [
        "radius error",
        "mean compactness",
        "concave points error",
        "worst compactness",
        "texture error",
        "concavity error",
        "mean fractal dimension"
    ],
    axis=1,
)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)

random_state=42
# 모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]
for model in models :
    #모델 구성
    
    #컴파일 , 훈련
    model.fit(x_train, y_train)

    #평가 예측
    from sklearn.metrics import accuracy_score
    acc = model.score(x_test, y_test)
    x_predict = model.predict(x_test)
    acc_pred = accuracy_score(y_test, x_predict)

    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)
    # print(type(model).__name__ ,":", model.feature_importances_)
    
    #25%이하 컬럼
    feature_importance_set = pd.DataFrame({'feature': x.columns, 'importance':model.feature_importances_})
    feature_importance_set.sort_values('importance', inplace=True)
    delete_0_25_features = feature_importance_set['feature'][:int(len(feature_importance_set.values) * 0.25)]
    delete_0_25_importance = feature_importance_set['importance'][:int(len(feature_importance_set.values) * 0.25)]
    print(f'''
    제거 할 컬럼명 : 
    {delete_0_25_features}  
    제거 할 feature_importances_ : 
    {delete_0_25_importance}    
    ''')


'''
[XGBClassifier] model acc :  0.9415204678362573
[XGBClassifier] eval_acc :  0.9415204678362573

제거후==============
[XGBClassifier] model acc :  0.9415204678362573
[XGBClassifier] eval_acc :  0.9415204678362573
'''
