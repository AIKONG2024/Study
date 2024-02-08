import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_iris(return_X_y=True)

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
acc : {scores} || 평균 acc : {round(np.mean(scores),4)}
eval acc : // {acc_score}
        """)
    except:
        continue
    
print("="*60)
print("[The Best score] : ", best_score )
print("[The Best model] : ", best_model )
print("="*60)

'''
============================================================
[The Best score] :  1.0
[The Best model] :  AdaBoostClassifier
============================================================
[kfold 적용 후]
acc :  [0.9        0.96666667 0.93333333 0.93333333 0.86666667]
평균 acc : 0.92
============================================================
[Stratifiedkfold 적용 후]
acc :  [0.96666667 1.         0.9        0.9        0.9       ]
평균 acc : 0.9333
============================================================
[스케일링 + train_test_split]
acc :  [0.91666667 0.91666667 1.         0.91666667 0.91666667]
평균 acc : 0.9333
eval acc_ : 0.9


=========================================================
[AdaBoostClassifier]
acc : [0.91666667 0.91666667 1.         0.91666667 0.91666667] || 평균 acc : 0.9333
eval acc : // 0.9666666666666667


=========================================================
[BaggingClassifier]
acc : [1.         0.91666667 1.         0.95833333 0.91666667] || 평균 acc : 0.9583
eval acc : // 0.9666666666666667


=========================================================
[BernoulliNB]
acc : [0.41666667 0.33333333 0.375      0.375      0.375     ] || 평균 acc : 0.375
eval acc : // 0.3333333333333333


=========================================================
[CalibratedClassifierCV]
acc : [0.75       0.91666667 0.91666667 0.95833333 0.83333333] || 평균 acc : 0.875
eval acc : // 0.8333333333333334


=========================================================
[DecisionTreeClassifier]
acc : [0.875      0.91666667 1.         0.95833333 0.91666667] || 평균 acc : 0.9333
eval acc : // 0.9


=========================================================
[DummyClassifier]
acc : [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333] || 평균 acc : 0.3333
eval acc : // 0.3333333333333333


=========================================================
[ExtraTreeClassifier]
acc : [0.95833333 0.875      0.91666667 0.875      0.95833333] || 평균 acc : 0.9167
eval acc : // 0.9


=========================================================
[ExtraTreesClassifier]
acc : [0.91666667 0.95833333 1.         0.95833333 0.91666667] || 평균 acc : 0.95
eval acc : // 0.9666666666666667


=========================================================
[GaussianNB]
acc : [0.91666667 0.91666667 1.         0.95833333 0.91666667] || 평균 acc : 0.9417
eval acc : // 0.9666666666666667


=========================================================
[GaussianProcessClassifier]
acc : [0.83333333 0.91666667 0.95833333 0.95833333 0.91666667] || 평균 acc : 0.9167
eval acc : // 0.8333333333333334


=========================================================
[GradientBoostingClassifier]
acc : [0.875      0.91666667 1.         0.91666667 0.91666667] || 평균 acc : 0.925
eval acc : // 0.8333333333333334


=========================================================
[HistGradientBoostingClassifier]
acc : [0.95833333 0.91666667 1.         0.95833333 0.91666667] || 평균 acc : 0.95
eval acc : // 0.3333333333333333


=========================================================
[KNeighborsClassifier]
acc : [0.91666667 1.         1.         0.95833333 0.95833333] || 평균 acc : 0.9667
eval acc : // 0.9666666666666667


=========================================================
[LabelPropagation]
acc : [0.875      0.95833333 1.         0.95833333 0.95833333] || 평균 acc : 0.95
eval acc : // 0.9666666666666667


=========================================================
[LabelSpreading]
acc : [0.875      0.95833333 1.         0.95833333 1.        ] || 평균 acc : 0.9583
eval acc : // 0.9666666666666667


=========================================================
[LinearDiscriminantAnalysis]
acc : [0.91666667 1.         1.         1.         0.91666667] || 평균 acc : 0.9667
eval acc : // 0.9666666666666667


=========================================================
[LinearSVC]
acc : [0.75       0.91666667 0.95833333 0.95833333 0.91666667] || 평균 acc : 0.9
eval acc : // 0.8666666666666667


=========================================================
[LogisticRegression]
acc : [0.83333333 0.91666667 0.95833333 0.95833333 0.91666667] || 평균 acc : 0.9167
eval acc : // 0.8666666666666667


=========================================================
[LogisticRegressionCV]
acc : [0.875      1.         1.         1.         0.91666667] || 평균 acc : 0.9583
eval acc : // 0.9666666666666667


=========================================================
[MLPClassifier]
acc : [0.91666667 0.91666667 1.         1.         0.95833333] || 평균 acc : 0.9583
eval acc : // 0.9333333333333333


=========================================================
[NearestCentroid]
acc : [0.875      0.91666667 1.         0.95833333 0.95833333] || 평균 acc : 0.9417
eval acc : // 0.8


=========================================================
[NuSVC]
acc : [0.875      1.         1.         0.95833333 0.91666667] || 평균 acc : 0.95
eval acc : // 0.9666666666666667


=========================================================
[PassiveAggressiveClassifier]
acc : [0.875      0.75       0.95833333 0.91666667 0.83333333] || 평균 acc : 0.8667
eval acc : // 0.8


=========================================================
[Perceptron]
acc : [0.83333333 0.75       0.95833333 0.91666667 0.66666667] || 평균 acc : 0.825
eval acc : // 0.7666666666666667


=========================================================
[QuadraticDiscriminantAnalysis]
acc : [0.95833333 1.         1.         1.         0.91666667] || 평균 acc : 0.975
eval acc : // 1.0


=========================================================
[RadiusNeighborsClassifier]
acc : [0.45833333 0.5        0.41666667 0.45833333 0.5       ] || 평균 acc : 0.4667
eval acc : // 0.6


=========================================================
[RandomForestClassifier]
acc : [0.875      0.95833333 1.         0.95833333 0.91666667] || 평균 acc : 0.9417
eval acc : // 0.9333333333333333


=========================================================
[RidgeClassifier]
acc : [0.70833333 0.875      0.875      0.95833333 0.79166667] || 평균 acc : 0.8417
eval acc : // 0.8


=========================================================
[RidgeClassifierCV]
acc : [0.625      0.875      0.83333333 0.875      0.83333333] || 평균 acc : 0.8083
eval acc : // 0.8


=========================================================
[SGDClassifier]
acc : [0.875      0.83333333 0.95833333 1.         0.79166667] || 평균 acc : 0.8917
eval acc : // 0.8333333333333334


=========================================================
[SVC]
acc : [0.91666667 1.         1.         1.         0.95833333] || 평균 acc : 0.975
eval acc : // 0.9

============================================================
[The Best score] :  1.0
[The Best model] :  QuadraticDiscriminantAnalysis
============================================================
'''
