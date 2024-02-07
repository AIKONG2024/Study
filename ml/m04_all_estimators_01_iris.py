from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = datasets = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=13, stratify=y) 

allAlgorithms = all_estimators(type_filter='classifier') #41개
# allAlgorithms = all_estimators(type_filter='regressor') #55개
best_acc = []

# 모델구성
for name, algorithm in allAlgorithms :
    try:
        # 모델
        model = algorithm()
        # 훈련
        model.fit(x_train, y_train)

        # 평가, 예측
        results = model.score(x_test, y_test)
        print(f"[{name}] score : ", results)
        x_predict = model.predict(x_test)
    except:
        continue

'''
[AdaBoostClassifier] score :  1.0
[BaggingClassifier] score :  1.0
[BernoulliNB] score :  0.3333333333333333
[CalibratedClassifierCV] score :  0.9666666666666667
[CategoricalNB] score :  0.9333333333333333
[ComplementNB] score :  0.6666666666666666
[DecisionTreeClassifier] score :  0.9666666666666667
[DummyClassifier] score :  0.3333333333333333
[ExtraTreeClassifier] score :  0.9666666666666667
[ExtraTreesClassifier] score :  0.9666666666666667
[GaussianNB] score :  1.0
[GaussianProcessClassifier] score :  1.0
[GradientBoostingClassifier] score :  1.0
[HistGradientBoostingClassifier] score :  0.9666666666666667
[KNeighborsClassifier] score :  0.9666666666666667
[LabelPropagation] score :  0.9666666666666667
[LabelSpreading] score :  0.9666666666666667
[LinearDiscriminantAnalysis] score :  1.0
[LinearSVC] score :  1.0
[LogisticRegression] score :  1.0
[LogisticRegressionCV] score :  1.0
[MLPClassifier] score :  0.9666666666666667
[MultinomialNB] score :  0.9666666666666667
[NearestCentroid] score :  0.9333333333333333
[NuSVC] score :  1.0
[PassiveAggressiveClassifier] score :  0.9333333333333333
[Perceptron] score :  0.6666666666666666
[QuadraticDiscriminantAnalysis] score :  1.0
[RadiusNeighborsClassifier] score :  1.0
[RandomForestClassifier] score :  1.0
[RidgeClassifier] score :  0.8666666666666667
[RidgeClassifierCV] score :  0.8666666666666667
[SGDClassifier] score :  0.9
[SVC] score :  1.0
    '''