# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1234567, stratify=y)

#모델 생성
allAlgorithms = all_estimators(type_filter='classifier') #55개
best_acc = 0
best_model = ""
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