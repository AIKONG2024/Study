from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

x,y = load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

allAlgorithms = all_estimators(type_filter='classifier')
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
[ARDRegression] score :  0.9019791193993927
[AdaBoostRegressor] score :  0.9326145552560647
[BaggingRegressor] score :  0.9305929919137467
[BayesianRidge] score :  0.9016707203852394
[DecisionTreeRegressor] score :  0.8652291105121294
[DummyRegressor] score :  -0.0010793845287229153
[ElasticNet] score :  -0.0010793845287229153
[ElasticNetCV] score :  0.9015280711692804
[ExtraTreeRegressor] score :  0.8652291105121294
[ExtraTreesRegressor] score :  0.9603571428571429
[GaussianProcessRegressor] score :  0.3516031579946851
[GradientBoostingRegressor] score :  0.9301925066639675
[HistGradientBoostingRegressor] score :  0.9269073674395104
[HuberRegressor] score :  0.8956074333137062
[KNeighborsRegressor] score :  0.9043126684636118
[KernelRidge] score :  -0.14888747448941886
[Lars] score :  0.900422427054671
[LarsCV] score :  0.9027367308984795
[Lasso] score :  -0.0010793845287229153
[LassoCV] score :  0.902720134751609
[LassoLars] score :  -0.0010793845287229153
[LassoLarsCV] score :  0.9027367308984795
[LassoLarsIC] score :  0.9035738068248668
[LinearRegression] score :  0.9004224270546709
[LinearSVR] score :  0.8807881980252816
[MLPRegressor] score :  0.8376528198340282
[NuSVR] score :  0.913301041514164
[OrthogonalMatchingPursuit] score :  0.7508160193458728
[OrthogonalMatchingPursuitCV] score :  0.8973362747678368
[PLSRegression] score :  0.8632558747099757
[PassiveAggressiveRegressor] score :  0.7093393674756092
[PoissonRegressor] score :  0.5254310468527558
[QuantileRegressor] score :  -0.010781671157530148
[RANSACRegressor] score :  0.8899295344616518
[RadiusNeighborsRegressor] score :  -6.836731300387083e+18
[RandomForestRegressor] score :  0.9368598382749327
[Ridge] score :  0.9017936284162703
[RidgeCV] score :  0.9017936284162923
[SGDRegressor] score :  0.8343319345889457
[SVR] score :  0.9102807156459127
[TheilSenRegressor] score :  0.8975415042338689
[TransformedTargetRegressor] score :  0.9004224270546709
[TweedieRegressor] score :  0.7152542419630801
====================================================================================================
[The Best score] :  0.9603571428571429
[The Best model] :  ExtraTreesRegressor
====================================================================================================
'''