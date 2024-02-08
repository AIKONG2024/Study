from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import numpy as np
import warnings
warnings.filterwarnings('ignore')

x,y = fetch_covtype(return_X_y=True)
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
acc : [0.51864203 0.34863708 0.51945956 0.53042103 0.59064554]
평균 acc : 0.5016
eval acc : // 0.5196767725445987


========================================================= 
[BernoulliNB]
acc : [0.63361373 0.63214001 0.63140853 0.63231213 0.63118942]
평균 acc : 0.6321
eval acc : // 0.631377847387761


=========================================================
[CalibratedClassifierCV]
acc : [0.71268906 0.7144317  0.71268906 0.71306555 0.71268597]
평균 acc : 0.7131

=========================================================
[CalibratedClassifierCV]
acc : [0.71268906 0.7144317  0.71268906 0.71306555 0.71268597]
평균 acc : 0.7131
eval acc : // 0.7108078104696092


=========================================================
[DecisionTreeClassifier]
acc : [0.93071362 0.93103634 0.93240249 0.93289731 0.9324663 ]
평균 acc : 0.9319
eval acc : // 0.8725334113577102


=========================================================
[DummyClassifier]
acc : [0.48759708 0.48759708 0.48759708 0.48759708 0.48760233]
평균 acc : 0.4876
eval acc : // 0.48760359026875383


=========================================================
[ExtraTreeClassifier]
acc : [0.84092425 0.84763667 0.86952733 0.84999247 0.85175504]
평균 acc : 0.852
eval acc : // 0.8029052606214986


=========================================================
[ExtraTreesClassifier]
acc : [0.94817237 0.94871023 0.94882855 0.94981821 0.94866665]
평균 acc : 0.9488
eval acc : // 0.913117561508739


=========================================================
[GaussianNB]
acc : [0.08925152 0.09089736 0.09445795 0.09012285 0.08981186]
평균 acc : 0.0909
eval acc : // 0.09215768956051049


=========================================================
[GradientBoostingClassifier]
acc : [0.77269207 0.77177772 0.77197134 0.77234784 0.77304461]
평균 acc : 0.7724
eval acc : // 0.7726478662340903


=========================================================
[HistGradientBoostingClassifier]
acc : [0.83721305 0.77810288 0.77878058 0.77906026 0.77739052]
평균 acc : 0.7901
eval acc : // 0.8207619424627591


=========================================================
[KNeighborsClassifier]
acc : [0.92934748 0.93069211 0.93045545 0.93101482 0.93006745]
평균 acc : 0.9303
eval acc : // 0.8815176888720601


=========================================================
[LinearDiscriminantAnalysis]
acc : [0.67820185 0.68088036 0.67894408 0.67914847 0.68231839]
평균 acc : 0.6799
eval acc : // 0.679070247756082


=========================================================
[LinearSVC]
acc : [0.71232331 0.71416278 0.71273208 0.71301177 0.71276127]
평균 acc : 0.713
eval acc : // 0.7108250217292152


=========================================================
[LogisticRegression]
acc : [0.71705643 0.72039113 0.71809987 0.71932618 0.71979647]
평균 acc : 0.7189
eval acc : // 0.7193445952342022
'''