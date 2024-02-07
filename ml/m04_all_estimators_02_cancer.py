import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)

allAlgorithms = all_estimators(type_filter='classifier') #41개
# allAlgorithms = all_estimators(type_filter='regressor') #55개
best_acc = 0
best_model = ""

# 모델구성
for name, algorithm in allAlgorithms :
    try:
        # 모델
        model = algorithm()
        # 훈련
        model.fit(x_train, y_train)

        # 평가, 예측
        results = model.score(x_test, y_test)
        if best_acc < results:
            best_acc = results
            best_model = name
        print(f"[{name}] score : ", results)
        x_predict = model.predict(x_test)
    except:
        continue
print("="*60)
print("[The Best score] : ", best_acc )
print("[The Best model] : ", best_model )
print("="*60)

'''
[AdaBoostClassifier] score :  0.9181286549707602
[BaggingClassifier] score :  0.9181286549707602
[BernoulliNB] score :  0.6140350877192983
[CalibratedClassifierCV] score :  0.9064327485380117
[ComplementNB] score :  0.8771929824561403
[DecisionTreeClassifier] score :  0.9064327485380117     
[DummyClassifier] score :  0.6140350877192983
[ExtraTreeClassifier] score :  0.9005847953216374        
[ExtraTreesClassifier] score :  0.935672514619883
[GaussianNB] score :  0.8888888888888888
[GaussianProcessClassifier] score :  0.9122807017543859
[GradientBoostingClassifier] score :  0.9239766081871345
[HistGradientBoostingClassifier] score :  0.9298245614035088
[KNeighborsClassifier] score :  0.935672514619883        
[LabelPropagation] score :  0.391812865497076
[LabelSpreading] score :  0.391812865497076
[LinearDiscriminantAnalysis] score :  0.9415204678362573 
[LinearSVC] score :  0.9064327485380117
[LogisticRegression] score :  0.9181286549707602
[LogisticRegressionCV] score :  0.9239766081871345
[MLPClassifier] score :  0.9415204678362573
[MultinomialNB] score :  0.8771929824561403
[NearestCentroid] score :  0.8713450292397661
[NuSVC] score :  0.847953216374269
[PassiveAggressiveClassifier] score :  0.8830409356725146
[Perceptron] score :  0.8304093567251462
[QuadraticDiscriminantAnalysis] score :  0.9415204678362573
[RandomForestClassifier] score :  0.9239766081871345
[RidgeClassifier] score :  0.935672514619883
[RidgeClassifierCV] score :  0.935672514619883
[SGDClassifier] score :  0.8830409356725146
[SVC] score :  0.8947368421052632
============================================================
[The Best score] :  0.9415204678362573
[The Best model] :  LinearDiscriminantAnalysis
============================================================
'''