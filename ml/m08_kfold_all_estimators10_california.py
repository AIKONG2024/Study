from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = fetch_california_housing(return_X_y=True)

#데이터 분류
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
from sklearn.metrics import r2_score
for name, algorithm in allAlgorithms :
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kf)
        y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
        acc_score = r2_score(y_test, y_predict)
        # 모델
        if best_score < acc_score:
            best_score = acc_score
            best_model = name
        # 3. 훈련
        # 평가, 예측
        print(f"""
=========================================================
[{name}]
r2 : {scores} 
평균 r2 : {round(np.mean(scores),4)}
eval r2 : // {acc_score}
        """)
    except:
        continue
    
print("="*60)
print("[The Best score] : ", best_score )
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