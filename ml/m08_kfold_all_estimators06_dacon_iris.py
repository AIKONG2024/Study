#https://dacon.io/competitions/open/236070/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

path = "C:/_data/dacon/iris/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

#데이터 확인
x = train_csv.drop(columns='species')
y = train_csv['species']

#데이터 분류
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
[AdaBoostClassifier] score :  1.0
[BaggingClassifier] score :  1.0
[BernoulliNB] score :  0.3333333333333333
[CalibratedClassifierCV] score :  0.9722222222222222
[CategoricalNB] score :  0.9444444444444444
[ComplementNB] score :  0.6666666666666666
[DecisionTreeClassifier] score :  1.0
[DummyClassifier] score :  0.3333333333333333
[ExtraTreeClassifier] score :  1.0
[ExtraTreesClassifier] score :  1.0
[GaussianNB] score :  1.0
[GaussianProcessClassifier] score :  1.0
[GradientBoostingClassifier] score :  1.0
[HistGradientBoostingClassifier] score :  1.0
[KNeighborsClassifier] score :  1.0
[LabelPropagation] score :  1.0
[LabelSpreading] score :  1.0
[LinearDiscriminantAnalysis] score :  1.0
[LinearSVC] score :  0.9722222222222222
[LogisticRegression] score :  1.0
[LogisticRegressionCV] score :  1.0
[MLPClassifier] score :  0.9722222222222222
[MultinomialNB] score :  0.8888888888888888
[NearestCentroid] score :  0.9444444444444444
[NuSVC] score :  1.0
[PassiveAggressiveClassifier] score :  0.9444444444444444
[Perceptron] score :  0.8333333333333334
[QuadraticDiscriminantAnalysis] score :  1.0
[RadiusNeighborsClassifier] score :  1.0
[RandomForestClassifier] score :  1.0
[RidgeClassifier] score :  0.8055555555555556
[RidgeClassifierCV] score :  0.8055555555555556
[SGDClassifier] score :  0.9166666666666666
[SVC] score :  1.0
============================================================
[The Best score] :  1.0
[The Best model] :  AdaBoostClassifier
============================================================
'''