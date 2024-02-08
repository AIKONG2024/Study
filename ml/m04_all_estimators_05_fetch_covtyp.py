from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

x,y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234, stratify=y)
print(x_test.shape)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# allAlgorithms = all_estimators(type_filter='classifier') #41개
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