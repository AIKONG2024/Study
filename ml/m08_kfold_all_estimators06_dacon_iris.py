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
=========================================================
[AdaBoostClassifier]
acc : [0.9        0.94736842 1.         1.         0.94736842]
평균 acc : 0.9589
eval acc : // 0.9166666666666666


=========================================================
[BaggingClassifier]
acc : [0.85       0.94736842 1.         1.         0.94736842]
평균 acc : 0.9489
eval acc : // 0.875


=========================================================
[BernoulliNB]
acc : [0.35       0.31578947 0.42105263 0.36842105 0.42105263]
평균 acc : 0.3753
eval acc : // 0.20833333333333334


=========================================================
[CalibratedClassifierCV]
acc : [0.8        0.84210526 0.89473684 0.89473684 0.94736842]
평균 acc : 0.8758
eval acc : // 0.75


=========================================================
[ComplementNB]
acc : [0.65       0.68421053 0.68421053 0.63157895 0.63157895]
평균 acc : 0.6563
eval acc : // 0.6666666666666666


=========================================================
[DecisionTreeClassifier]
acc : [0.85       0.94736842 1.         1.         1.        ]
평균 acc : 0.9595
eval acc : // 0.875


=========================================================
[DummyClassifier]
acc : [0.3        0.31578947 0.31578947 0.31578947 0.31578947]
평균 acc : 0.3126
eval acc : // 0.20833333333333334


=========================================================
[ExtraTreeClassifier]
acc : [0.85       0.94736842 0.94736842 1.         0.94736842]
평균 acc : 0.9384
eval acc : // 0.875


=========================================================
[ExtraTreesClassifier]
acc : [0.85       0.94736842 1.         1.         1.        ]
평균 acc : 0.9595
eval acc : // 0.9166666666666666


=========================================================
[GaussianNB]
acc : [0.9        0.94736842 0.89473684 0.94736842 1.        ]
평균 acc : 0.9379
eval acc : // 0.875


=========================================================
[GaussianProcessClassifier]
acc : [0.85       0.94736842 0.89473684 0.94736842 0.94736842]
평균 acc : 0.9174
eval acc : // 0.75


=========================================================
[GradientBoostingClassifier]
acc : [0.85       0.94736842 1.         1.         0.94736842]
평균 acc : 0.9489
eval acc : // 0.8333333333333334


=========================================================
[HistGradientBoostingClassifier]
acc : [0.85       0.94736842 1.         1.         0.89473684] 
평균 acc : 0.9384
eval acc : // 0.20833333333333334
        

=========================================================
[KNeighborsClassifier]
acc : [0.9        0.94736842 1.         0.94736842 0.94736842] 
평균 acc : 0.9484
eval acc : // 0.9166666666666666
        

=========================================================
[LabelPropagation]
acc : [0.85       0.94736842 1.         0.94736842 1.        ]
평균 acc : 0.9489
eval acc : // 0.9166666666666666


=========================================================
[LabelSpreading]
acc : [0.85       0.94736842 1.         0.94736842 1.        ]
평균 acc : 0.9489
eval acc : // 0.9166666666666666


=========================================================
[LinearDiscriminantAnalysis]
acc : [0.95 1.   1.   1.   1.  ]
평균 acc : 0.99
eval acc : // 0.9583333333333334


=========================================================
[LinearSVC]
acc : [0.85       0.89473684 0.89473684 0.94736842 0.94736842]
평균 acc : 0.9068
eval acc : // 0.75


=========================================================
[LogisticRegression]
acc : [0.85       0.89473684 0.89473684 0.94736842 0.94736842]
평균 acc : 0.9068
eval acc : // 0.75


=========================================================
[LogisticRegressionCV]
acc : [0.85       0.94736842 1.         1.         1.        ]
평균 acc : 0.9595
eval acc : // 0.9166666666666666


=========================================================
[MLPClassifier]
acc : [0.85       0.94736842 0.84210526 0.94736842 1.        ]
평균 acc : 0.9174
eval acc : // 0.9166666666666666


=========================================================
[MultinomialNB]
acc : [0.65       0.68421053 0.68421053 0.68421053 0.68421053]
평균 acc : 0.6774
eval acc : // 0.625


=========================================================
[NearestCentroid]
acc : [0.85       0.94736842 0.89473684 0.89473684 1.        ]
평균 acc : 0.9174
eval acc : // 0.9166666666666666


=========================================================
[NuSVC]
acc : [0.85       0.94736842 1.         0.94736842 1.        ]
평균 acc : 0.9489
eval acc : // 0.9583333333333334


=========================================================
[PassiveAggressiveClassifier]
acc : [0.85       0.84210526 1.         0.94736842 0.84210526]
평균 acc : 0.8963
eval acc : // 0.7083333333333334


=========================================================
[Perceptron]
acc : [0.65       0.78947368 0.94736842 0.94736842 0.89473684]
평균 acc : 0.8458
eval acc : // 0.5833333333333334


=========================================================
[QuadraticDiscriminantAnalysis]
acc : [0.9 1.  1.  1.  1. ]
평균 acc : 0.98
eval acc : // 0.7083333333333334


=========================================================
[RadiusNeighborsClassifier]
acc : [0.6        0.52631579 0.52631579 0.52631579 0.42105263]
평균 acc : 0.52
eval acc : // 0.4166666666666667


=========================================================
[RandomForestClassifier]
acc : [0.85       0.94736842 1.         1.         1.        ]
평균 acc : 0.9595
eval acc : // 0.9166666666666666


=========================================================
[RidgeClassifier]
acc : [0.8        0.78947368 0.84210526 0.84210526 0.94736842]
평균 acc : 0.8442
eval acc : // 0.625


=========================================================
[RidgeClassifierCV]
acc : [0.8        0.84210526 0.89473684 0.89473684 0.84210526]
평균 acc : 0.8547
eval acc : // 0.625


=========================================================
[SGDClassifier]
acc : [0.8        0.73684211 0.89473684 0.94736842 0.89473684]
평균 acc : 0.8547
eval acc : // 0.5833333333333334


=========================================================
[SVC]
acc : [0.85       0.94736842 1.         0.94736842 1.        ]
평균 acc : 0.9489
eval acc : // 0.9166666666666666

============================================================
[The Best score] :  0.9583333333333334
[The Best model] :  LinearDiscriminantAnalysis
============================================================
'''