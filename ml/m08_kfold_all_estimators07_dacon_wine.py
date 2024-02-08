# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
allAlgorithms = all_estimators(type_filter='classifier') #41개
# allAlgorithms = all_estimators(type_filter='regressor') #55개
best_score = 0
best_model = ""

# 모델구성
from sklearn.metrics import accuracy_score
for name, algorithm in allAlgorithms :
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kf)
        y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
        acc_score = accuracy_score(y_test, y_predict)
        # 모델
        if best_score < acc_score:
            best_score = acc_score
            best_model = name
        # 3. 훈련
        # 평가, 예측
        print(f"""
=========================================================
[{name}]
acc : {scores} 
평균 acc : {round(np.mean(scores),4)}
eval acc : // {acc_score}
        """)
    except:
        continue
    
print("="*60)
print("[The Best score] : ", best_score )
print("[The Best model] : ", best_model )
print("="*60)
'''
[AdaBoostClassifier] score :  0.3406060606060606
[BaggingClassifier] score :  0.6327272727272727
[BernoulliNB] score :  0.44363636363636366
[CalibratedClassifierCV] score :  0.49818181818181817
[ComplementNB] score :  0.31393939393939396
[DecisionTreeClassifier] score :  0.5793939393939394
[DummyClassifier] score :  0.4387878787878788
[ExtraTreeClassifier] score :  0.5721212121212121        
[ExtraTreesClassifier] score :  0.6472727272727272
[GaussianNB] score :  0.3890909090909091
[GaussianProcessClassifier] score :  0.5757575757575758
[GradientBoostingClassifier] score :  0.5672727272727273
[HistGradientBoostingClassifier] score :  0.6254545454545455
[KNeighborsClassifier] score :  0.48727272727272725
[LabelPropagation] score :  0.5696969696969697
[LabelSpreading] score :  0.5709090909090909
[LinearDiscriminantAnalysis] score :  0.5381818181818182
[LinearSVC] score :  0.45454545454545453
[LogisticRegression] score :  0.46545454545454545
[LogisticRegressionCV] score :  0.5212121212121212
[MLPClassifier] score :  0.5187878787878788
[MultinomialNB] score :  0.3527272727272727
[NearestCentroid] score :  0.13454545454545455
[PassiveAggressiveClassifier] score :  0.3321212121212121
[Perceptron] score :  0.05333333333333334
[QuadraticDiscriminantAnalysis] score :  0.48848484848484847
[RandomForestClassifier] score :  0.6448484848484849
[RidgeClassifier] score :  0.5333333333333333
[RidgeClassifierCV] score :  0.5333333333333333
[SGDClassifier] score :  0.3054545454545455
[SVC] score :  0.44242424242424244
============================================================
[The Best score] :  0.6472727272727272
[The Best model] :  ExtraTreesClassifier
============================================================
'''