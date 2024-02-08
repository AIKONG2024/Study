from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
import numpy as np
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

x,y = load_wine(return_X_y=True)
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
acc : [0.82758621 0.93103448 0.85714286 0.85714286 0.78571429]
평균 acc : 0.8517
eval acc : // 0.9166666666666666


========================================================= 
[BaggingClassifier]
acc : [0.93103448 0.96551724 0.96428571 0.96428571 0.92857143]
평균 acc : 0.9507
eval acc : // 0.9444444444444444


========================================================= 
[BernoulliNB]
acc : [0.34482759 0.34482759 0.32142857 0.39285714 0.39285714]
평균 acc : 0.3594
eval acc : // 0.3888888888888889


========================================================= 
[CalibratedClassifierCV]
acc : [1.         0.96551724 0.92857143 1.         1.        ]
평균 acc : 0.9788
eval acc : // 1.0


========================================================= 
[DecisionTreeClassifier]
acc : [0.93103448 0.89655172 0.92857143 0.96428571 0.89285714]
평균 acc : 0.9227
eval acc : // 0.9444444444444444


========================================================= 
[DummyClassifier]
acc : [0.4137931  0.4137931  0.39285714 0.39285714 0.39285714]
평균 acc : 0.4012
eval acc : // 0.3888888888888889


========================================================= 
[ExtraTreeClassifier]
acc : [0.86206897 0.89655172 0.75       0.92857143 0.89285714]
평균 acc : 0.866
eval acc : // 0.7777777777777778


========================================================= 
[ExtraTreesClassifier]
acc : [0.96551724 0.93103448 1.         1.         1.        ]
평균 acc : 0.9793
eval acc : // 0.9166666666666666


========================================================= 
[GaussianNB]
acc : [1.         0.96551724 0.96428571 0.96428571 0.92857143]
평균 acc : 0.9645
eval acc : // 0.9722222222222222


========================================================= 
[GaussianProcessClassifier]
acc : [1.         0.93103448 1.         1.         1.        ]
평균 acc : 0.9862
eval acc : // 0.9722222222222222


========================================================= 
[GradientBoostingClassifier]
acc : [0.89655172 0.96551724 0.89285714 0.92857143 0.96428571]
평균 acc : 0.9296
eval acc : // 0.9166666666666666


========================================================= 
[HistGradientBoostingClassifier]
acc : [0.96551724 0.89655172 1.         0.96428571 1.        ]
평균 acc : 0.9653
eval acc : // 0.3888888888888889


========================================================= 
[KNeighborsClassifier]
acc : [0.96551724 0.89655172 1.         1.         1.        ]
평균 acc : 0.9724
eval acc : // 0.9722222222222222


========================================================= 
[LabelPropagation]
acc : [0.93103448 0.93103448 0.96428571 1.         1.        ]
평균 acc : 0.9653
eval acc : // 0.9444444444444444


========================================================= 
[LabelSpreading]
acc : [0.93103448 0.93103448 0.96428571 1.         1.        ]
평균 acc : 0.9653
eval acc : // 0.9444444444444444


========================================================= 
[LinearDiscriminantAnalysis]
acc : [1.         1.         1.         0.96428571 1.        ]
평균 acc : 0.9929
eval acc : // 0.9444444444444444


========================================================= 
[LinearSVC]
acc : [1.         0.89655172 0.96428571 1.         1.        ]
평균 acc : 0.9722
eval acc : // 0.9722222222222222


========================================================= 
[LogisticRegression]
acc : [1.         0.96551724 1.         1.         1.        ]
평균 acc : 0.9931
eval acc : // 0.9722222222222222


========================================================= 
[LogisticRegressionCV]
acc : [1.         0.96551724 0.96428571 1.         1.        ]
평균 acc : 0.986
eval acc : // 0.9444444444444444


========================================================= 
[MLPClassifier]
acc : [1.         0.93103448 0.92857143 1.         1.        ]
평균 acc : 0.9719
eval acc : // 0.9722222222222222


========================================================= 
[NearestCentroid]
acc : [0.86206897 0.93103448 0.96428571 1.         0.96428571]
평균 acc : 0.9443
eval acc : // 0.9722222222222222


========================================================= 
[NuSVC]
acc : [1.         0.93103448 0.96428571 0.96428571 0.96428571]
평균 acc : 0.9648
eval acc : // 1.0


========================================================= 
[PassiveAggressiveClassifier]
acc : [1.         0.93103448 0.96428571 0.96428571 1.        ]
평균 acc : 0.9719
eval acc : // 1.0


========================================================= 
[Perceptron]
acc : [1.         1.         0.92857143 1.         1.        ]
평균 acc : 0.9857
eval acc : // 0.9722222222222222


========================================================= 
[QuadraticDiscriminantAnalysis]
acc : [1.         0.96551724 0.96428571 1.         0.96428571]
평균 acc : 0.9788
eval acc : // 0.4722222222222222


========================================================= 
[RadiusNeighborsClassifier]
acc : [0.89655172 0.96551724 0.89285714 1.         1.        ]
평균 acc : 0.951
eval acc : // 0.9166666666666666


========================================================= 
[RandomForestClassifier]
acc : [0.96551724 0.96551724 1.         1.         1.        ]
평균 acc : 0.9862
eval acc : // 0.9722222222222222


========================================================= 
[RidgeClassifier]
acc : [1.         0.96551724 1.         1.         1.        ]
평균 acc : 0.9931
eval acc : // 0.9444444444444444


========================================================= 
[RidgeClassifierCV]
acc : [1. 1. 1. 1. 1.]
평균 acc : 1.0
eval acc : // 0.9444444444444444


========================================================= 
[SGDClassifier]
acc : [1.         0.96551724 0.89285714 1.         1.        ]
평균 acc : 0.9717
eval acc : // 1.0


========================================================= 
[SVC]
acc : [1.         0.93103448 0.92857143 0.96428571 1.        ]
평균 acc : 0.9648
eval acc : // 1.0

============================================================
[The Best score] :  1.0
[The Best model] :  CalibratedClassifierCV
============================================================
'''