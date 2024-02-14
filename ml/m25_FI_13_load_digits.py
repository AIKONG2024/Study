from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time

datasets = load_digits()
x= datasets.data
y= datasets.target

# print(x.shape, y.shape) #(1797, 64) (1797,)
# print(pd.value_counts(y, sort=False))

# 넘파이로 삭제
# x = np.delete(x, 0, axis=1)
# 판다스로 삭제
x = pd.DataFrame(data=x, columns=datasets.feature_names)
x = x.drop(
    [
        "pixel_0_0",
        "pixel_4_7",
        "pixel_3_7",
        "pixel_7_0",
        "pixel_5_0",
        "pixel_1_0",
        "pixel_3_0",
        "pixel_6_0",
        "pixel_2_0",
        "pixel_5_7",
        "pixel_2_7",
        "pixel_1_7",
        "pixel_0_7",
        "pixel_6_7",
        "pixel_7_1",
    ],
    axis=1,
)

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
[XGBClassifier] model acc :  0.975
[XGBClassifier] eval_acc :  0.975
======================
제거후
[XGBClassifier] model acc :  0.9722222222222222
[XGBClassifier] eval_acc :  0.9722222222222222
'''