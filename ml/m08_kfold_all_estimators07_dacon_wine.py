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
=========================================================
[AdaBoostClassifier]
acc : [0.47840909 0.44204545 0.39817975 0.43003413 0.40045506] 
평균 acc : 0.4298
eval acc : // 0.46454545454545454
        

=========================================================
[BaggingClassifier]
acc : [0.6125     0.6375     0.61547213 0.62116041 0.60864619] 
평균 acc : 0.6191
eval acc : // 0.5254545454545455


=========================================================
[BernoulliNB]
acc : [0.43181818 0.44431818 0.4334471  0.44482366 0.41638225]
평균 acc : 0.4342
eval acc : // 0.4690909090909091


=========================================================
[CalibratedClassifierCV]
acc : [0.51363636 0.52613636 0.54948805 0.56200228 0.54152446]
평균 acc : 0.5386
eval acc : // 0.5390909090909091


=========================================================
[DecisionTreeClassifier]
acc : [0.55340909 0.55568182 0.53924915 0.57337884 0.55062571]
평균 acc : 0.5545
eval acc : // 0.46636363636363637


=========================================================
[DummyClassifier]
acc : [0.43977273 0.43977273 0.43913538 0.43913538 0.43913538]
평균 acc : 0.4394
eval acc : // 0.44


=========================================================
[ExtraTreeClassifier]
acc : [0.5375     0.54886364 0.54948805 0.57679181 0.55517634]
평균 acc : 0.5536
eval acc : // 0.4727272727272727


=========================================================
[ExtraTreesClassifier]
acc : [0.63181818 0.65227273 0.63253697 0.65756542 0.66097838]
평균 acc : 0.647
eval acc : // 0.55


=========================================================
[GaussianNB]
acc : [0.40568182 0.43409091 0.407281   0.4254835  0.41410694]
평균 acc : 0.4173
eval acc : // 0.38545454545454544


=========================================================
[GaussianProcessClassifier]
acc : [0.5125     0.54318182 0.54152446 0.55517634 0.53469852]
평균 acc : 0.5374
eval acc : // 0.5290909090909091


=========================================================
[GradientBoostingClassifier]
acc : [0.56818182 0.56704545 0.5847554  0.58703072 0.592719  ]
평균 acc : 0.5799
eval acc : // 0.5627272727272727


=========================================================
[HistGradientBoostingClassifier]
acc : [0.62613636 0.64318182 0.63481229 0.63594994 0.62116041]
평균 acc : 0.6322
eval acc : // 0.5454545454545454


=========================================================
[KNeighborsClassifier]
acc : [0.52840909 0.56931818 0.55176337 0.53924915 0.55290102]
평균 acc : 0.5483
eval acc : // 0.5063636363636363


=========================================================
[LabelPropagation]
acc : [0.49659091 0.52954545 0.53242321 0.53583618 0.51422071]
평균 acc : 0.5217
eval acc : // 0.509090909090909


=========================================================
[LabelSpreading]
acc : [0.49204545 0.53295455 0.52901024 0.52787258 0.50853242] 
평균 acc : 0.5181
eval acc : // 0.5045454545454545
        

=========================================================
[LinearDiscriminantAnalysis]
acc : [0.50909091 0.52954545 0.54607509 0.56541524 0.54266212] 
평균 acc : 0.5386
eval acc : // 0.5390909090909091
        

=========================================================
[LinearSVC]
acc : [0.50795455 0.53409091 0.53924915 0.55631399 0.53469852] 
평균 acc : 0.5345
eval acc : // 0.5381818181818182


=========================================================
[LogisticRegression]
acc : [0.51136364 0.53295455 0.54379977 0.55517634 0.5403868 ]
평균 acc : 0.5367
eval acc : // 0.5345454545454545


=========================================================
[MLPClassifier]
acc : [0.53181818 0.54772727 0.55176337 0.56541524 0.5745165 ]
평균 acc : 0.5542
eval acc : // 0.5454545454545454


=========================================================
[NearestCentroid]
acc : [0.08636364 0.14659091 0.19112628 0.15585893 0.13538111]
평균 acc : 0.1431
eval acc : // 0.2018181818181818


=========================================================
[PassiveAggressiveClassifier]
acc : [0.3375     0.09545455 0.52901024 0.41069397 0.40045506]
평균 acc : 0.3546
eval acc : // 0.47454545454545455


=========================================================
[Perceptron]
acc : [0.39204545 0.27613636 0.43572241 0.37883959 0.33333333]
평균 acc : 0.3632
eval acc : // 0.48454545454545456


=========================================================
[RadiusNeighborsClassifier]
acc : [       nan 0.43977273 0.44254835 0.41183163 0.41865757]
평균 acc : nan
eval acc : // 0.4672727272727273


=========================================================
[RandomForestClassifier]
acc : [0.63977273 0.65       0.63936291 0.66097838 0.65642776]
평균 acc : 0.6493
eval acc : // 0.5645454545454546


=========================================================
[RidgeClassifier]
acc : [0.51022727 0.52954545 0.5403868  0.55745165 0.53242321]
평균 acc : 0.534
eval acc : // 0.5327272727272727


=========================================================
[RidgeClassifierCV]
acc : [0.50795455 0.53068182 0.53469852 0.55745165 0.53697383]
평균 acc : 0.5336
eval acc : // 0.5327272727272727


=========================================================
[SGDClassifier]
acc : [0.50568182 0.50568182 0.49943117 0.51877133 0.5301479 ]
평균 acc : 0.5119
eval acc : // 0.5209090909090909


=========================================================
[SVC]
acc : [0.51477273 0.54090909 0.53924915 0.55176337 0.53469852]
평균 acc : 0.5363
eval acc : // 0.53

============================================================
[The Best score] :  0.5645454545454546
[The Best model] :  RandomForestClassifier
============================================================
'''