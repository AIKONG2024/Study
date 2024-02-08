from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = fetch_california_housing(return_X_y=True)

# 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, random_state=2874458)

# 모델 구성
allAlgorithms = all_estimators(type_filter='regressor') #55개
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
[ARDRegression] score :  0.60126889518111
[AdaBoostRegressor] score :  0.37485270522063885
[BaggingRegressor] score :  0.781965730659791
[BayesianRidge] score :  0.6133578073814252
[DecisionTreeRegressor] score :  0.6024862720271628
[DummyRegressor] score :  -2.004208295702803e-05
[ElasticNet] score :  0.42722686995606785
[ElasticNetCV] score :  0.5946323560030056
[ExtraTreeRegressor] score :  0.5281890753492615
[ExtraTreesRegressor] score :  0.8146634051035847
[GammaRegressor] score :  -2.0234608472335935e-05
[ExtraTreesRegressor] score :  0.8146634051035847
[GammaRegressor] score :  -2.0234608472335935e-05
[GaussianProcessRegressor] score :  -2.8118307153020923
[GradientBoostingRegressor] score :  0.7894213579770931
[HistGradientBoostingRegressor] score :  0.8341553976013417
[HuberRegressor] score :  0.5346626656815499
[KNeighborsRegressor] score :  0.15627721317983434
[KernelRidge] score :  0.5614270533769874
[Lars] score :  0.613385606397415
[LarsCV] score :  0.6131180513547954
[Lasso] score :  0.2844436894123942
[LassoCV] score :  0.5978292767842728
[LassoLars] score :  -2.004208295702803e-05
[LassoLarsCV] score :  0.6131180513547954
[LassoLarsIC] score :  0.613385606397415
[LinearRegression] score :  0.6133856063974148
[LinearSVR] score :  -1.1621090041994573
[MLPRegressor] score :  0.2800651859120129
[NuSVR] score :  0.0058555613537981666
[OrthogonalMatchingPursuit] score :  0.4874239905223867
[OrthogonalMatchingPursuitCV] score :  0.6049279745906894[PLSRegression] score :  0.5421660653258545
[PassiveAggressiveRegressor] score :  -0.7925680208374257[PoissonRegressor] score :  0.46799514405163734
[PLSRegression] score :  0.5421660653258545
[PassiveAggressiveRegressor] score :  0.420937285465281
[PoissonRegressor] score :  0.46799514405163734
'''