from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time

x,y = load_digits(return_X_y=True)

print(x.shape, y.shape) #(1797, 64) (1797,)
print(pd.value_counts(y, sort=False))

#Random으로 1번만 돌리고
#Grid Search, Randomized Search 로 돌려보기
#시간체크

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]

for model in models :
    model.fit(x_train, y_train)

        #예측 평가
    acc = model.score(x_test, y_test)
    y_predict = model.predict(x_test)

    acc_pred = accuracy_score(y_test, y_predict)
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)
    print(type(model).__name__ ,":", model.feature_importances_)

'''

'''