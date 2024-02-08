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
[AdaBoostClassifier] score :  0.470339177528915
[BaggingClassifier] score :  0.9593870479162842
[BernoulliNB] score :  0.6597668441343859
[CalibratedClassifierCV] score :  0.7118654764090325
[DecisionTreeClassifier] score :  0.9351305764641087
[DummyClassifier] score :  0.487602120433266
[ExtraTreeClassifier] score :  0.8581042316871672
[ExtraTreesClassifier] score :  0.9514124747567468
[GaussianNB] score :  0.09477120433266017
[GradientBoostingClassifier] score :  0.773304112355425
[HistGradientBoostingClassifier] score :  0.8098838810354323
[KNeighborsClassifier] score :  0.9258249954103176
[LinearDiscriminantAnalysis] score :  0.6791582522489443
[LinearSVC] score :  0.7117908940701303
[LogisticRegression] score :  0.7229495593904902
[LogisticRegressionCV] score :  0.7234716357628053
[MLPClassifier] score :  0.8758605654488709
[NearestCentroid] score :  0.44815380025702223
[PassiveAggressiveClassifier] score :  0.566538920506701
[Perceptron] score :  0.5994354690655407
[QuadraticDiscriminantAnalysis] score :  0.08755966587112173
[RandomForestClassifier] score :  0.9531106572425189
[RidgeClassifier] score :  0.7000757297595006
[RidgeClassifierCV] score :  0.7000872039654856
[SGDClassifier] score :  0.7104943087938315
[SVC] score :  0.8295105103726822
============================================================
[The Best score] :  0.9593870479162842
[The Best model] :  BaggingClassifier
'''