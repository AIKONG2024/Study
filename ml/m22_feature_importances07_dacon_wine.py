# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
y = lbe.fit_transform(y)

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1234567, stratify=y)


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
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)
    print(type(model).__name__ ,":", model.feature_importances_)
    
    submission_csv['quality'] =  lbe.inverse_transform(submission)
    submission_csv['quality'] += 3


    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)
'''
[DecisionTreeClassifier] model acc :  0.5757575757575758
[DecisionTreeClassifier] eval_acc :  0.5757575757575758
DecisionTreeClassifier : [0.09014257 0.11175784 0.07191506 0.08474466 0.08934456 0.07085378
 0.0873578  0.0766169  0.08253168 0.08413444 0.15060071 0.        ]
[RandomForestClassifier] model acc :  0.6545454545454545
[RandomForestClassifier] eval_acc :  0.6545454545454545
RandomForestClassifier : [0.07509722 0.09986209 0.08073238 0.08342594 0.08880845 0.08515029
 0.0889254  0.10005155 0.08287804 0.08797225 0.12328552 0.00381087]
[GradientBoostingClassifier] model acc :  0.5684848484848485
[GradientBoostingClassifier] eval_acc :  0.5684848484848485
GradientBoostingClassifier : [0.03530588 0.14763536 0.05337533 0.07366852 0.05841559 0.05754826
 0.05769353 0.06178558 0.05742395 0.08013112 0.31156771 0.00544917]
[XGBClassifier] model acc :  0.6266666666666667
[XGBClassifier] eval_acc :  0.6266666666666667
XGBClassifier : [0.05479925 0.08467108 0.06196449 0.06593997 0.05291978 0.06464206
 0.06005028 0.05360193 0.05742282 0.06102384 0.16808999 0.21487454]
'''

