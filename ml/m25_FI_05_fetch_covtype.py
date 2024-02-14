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
import pandas as pd

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# 넘파이로 삭제
# x = np.delete(x, 0, axis=1)
# 판다스로 삭제
x = pd.DataFrame(data=x, columns=datasets.feature_names)
x = x.drop(
    [
        "Soil_Type_14",
        "Soil_Type_6",
        "Soil_Type_7",
        "Soil_Type_35",
        "Soil_Type_8",
        "Soil_Type_24",
        "Soil_Type_27",
        "Soil_Type_17",
        "Soil_Type_25",
        "Soil_Type_13",
        "Soil_Type_4",
        "Soil_Type_36",
        "Soil_Type_33",
    ],
    axis=1,
)


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
    # print(type(model).__name__ ,":", model.feature_importances_)
    
    # 25%이하 컬럼
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
[RandomForestClassifier] model acc :  0.9532081879933909
[RandomForestClassifier] eval_acc :  0.9532081879933909
=================
제거후
[RandomForestClassifier] model acc :  0.9542982375619607
[RandomForestClassifier] eval_acc :  0.012541307141545804
'''