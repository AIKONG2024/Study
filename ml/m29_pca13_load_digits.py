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

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
random_state=42

# 모델 구성
model = RandomForestClassifier()
for idx in range(1,len(datasets.feature_names)) :
    model.fit(x_train, y_train)

    # 평가 예측
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    predict = model.predict(x_test)
    print(f'''
    pca n_components : {idx} 
    score : {accuracy_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')
'''
    pca n_components : 1
    score : 0.9888888888888889
    걸린 시간 : 0.16 초


    pca n_components : 2
    score : 0.9944444444444445
    걸린 시간 : 0.16 초


    pca n_components : 3
    score : 0.9861111111111112
    걸린 시간 : 0.15 초


    pca n_components : 4
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 5
    score : 0.9972222222222222
    걸린 시간 : 0.15 초


    pca n_components : 6
    score : 0.9888888888888889
    걸린 시간 : 0.16 초


    pca n_components : 7
    score : 0.9861111111111112
    걸린 시간 : 0.16 초


    pca n_components : 8
    score : 0.9916666666666667
    걸린 시간 : 0.16 초


    pca n_components : 9
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 10
    score : 0.9861111111111112
    걸린 시간 : 0.15 초


    pca n_components : 11
    score : 0.9916666666666667
    걸린 시간 : 0.16 초


    pca n_components : 12
    score : 0.9861111111111112
    걸린 시간 : 0.15 초


    pca n_components : 13
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 14
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 15
    score : 0.9861111111111112
    걸린 시간 : 0.15 초


    pca n_components : 16
    score : 0.9861111111111112
    걸린 시간 : 0.15 초


    pca n_components : 17
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 18
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 19
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 20
    score : 0.9916666666666667
    걸린 시간 : 0.16 초


    pca n_components : 21
    score : 0.9916666666666667
    걸린 시간 : 0.16 초


    pca n_components : 22
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 23
    score : 0.9916666666666667
    걸린 시간 : 0.17 초


    pca n_components : 24 
    score : 0.9888888888888889
    걸린 시간 : 0.16 초
    

    pca n_components : 25 
    score : 0.9888888888888889
    걸린 시간 : 0.15 초
    

    pca n_components : 26 
    score : 0.9916666666666667
    걸린 시간 : 0.15 초
    

    pca n_components : 27
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 28
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 29
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 30
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 31
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 32
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 33
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 34
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 35
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 36
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 37
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 38
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 39
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 40
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 41
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 42
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 43
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 44
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 45
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 46
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 47
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 48
    score : 0.9944444444444445
    걸린 시간 : 0.16 초


    pca n_components : 49
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 50
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 51
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 52
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 53
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 54
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 55
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 56
    score : 0.9916666666666667
    걸린 시간 : 0.16 초


    pca n_components : 57
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 58
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 59
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 60
    score : 0.9916666666666667
    걸린 시간 : 0.15 초


    pca n_components : 61
    score : 0.9944444444444445
    걸린 시간 : 0.15 초


    pca n_components : 62
    score : 0.9888888888888889
    걸린 시간 : 0.15 초


    pca n_components : 63
    score : 0.9888888888888889
    걸린 시간 : 0.15 초
'''