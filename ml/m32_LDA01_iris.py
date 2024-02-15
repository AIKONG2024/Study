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
import time
from sklearn.metrics import accuracy_score

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

model = RandomForestClassifier()
unique = np.unique(datasets.target)
# print(unique)
for idx in range(1,min(len(datasets.feature_names), len(unique))) :

    # 컴파일, 훈련
    model.fit(x_train, y_train)

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
    score : 1.0
    걸린 시간 : 0.05 초


    pca n_components : 2
    score : 0.9666666666666667
    걸린 시간 : 0.05 초
'''
    
