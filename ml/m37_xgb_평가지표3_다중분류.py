from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
seed = 42

datasets = load_digits()
x= datasets.data
y= datasets.target

X_train, X_test, y_train , y_test = train_test_split(
x, y, shuffle= True, random_state=123, train_size=0.8,
stratify= y
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {
    'n_estimators': 4000,  # 디폴트 100
    'learning_rate': 0.01,  # 디폴트 0.3 / 0~1 / eta *
    'max_depth': 2,  # 디폴트 0 / 0~inf
    'early_stopping_rounds' : 100
}

# 2. 모델
model = XGBClassifier(random_state = seed, **parameters)
model.fit(X_train, y_train, eval_set = [(X_test, y_test)], eval_metric = "merror")
#분류 : mlogloss, merror, auc (이진이 더 좋음)  // f1 score: mlogloss
#문서 : https://xgboost.readthedocs.io/en/stable/parameter.html
print("파라미터 : ",model.get_params())
results = model.score(X_test, y_test)
x_predictsion = model.predict(X_test)
best_score = accuracy_score(y_test, x_predictsion) 
print(
f"""
============================================
[Best acc_score: {best_score}]
============================================
"""
)
