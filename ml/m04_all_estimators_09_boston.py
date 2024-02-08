import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

datasets= load_boston()
x = datasets.data
y = datasets.target

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=20)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구성
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
[ARDRegression] score :  0.680223386023765
[AdaBoostRegressor] score :  0.8200252314585107
[BaggingRegressor] score :  0.5942744678240062
[BayesianRidge] score :  0.7019993598356585
[DecisionTreeRegressor] score :  0.6504443058195355      
[DummyRegressor] score :  -0.0029983028238687037
[ElasticNet] score :  0.14907165955963597
[ElasticNetCV] score :  0.6978705613989958
[ExtraTreeRegressor] score :  0.5676099166481212
[ExtraTreesRegressor] score :  0.8612048082268853
[GammaRegressor] score :  0.19108497986833894
[GaussianProcessRegressor] score :  -0.32975288374670075
[GradientBoostingRegressor] score :  0.8406316030450138
[HistGradientBoostingRegressor] score :  0.7977087476985938
[HuberRegressor] score :  0.6646847358539809
[KNeighborsRegressor] score :  0.6943369094988743        
[KernelRidge] score :  0.5943609599200794
[Lars] score :  0.7023964981707981
[LarsCV] score :  0.6639122910722697
[Lasso] score :  0.23481680492080725
[LassoCV] score :  0.7013024877380519
[LassoLars] score :  -0.0029983028238687037
[LassoLarsCV] score :  0.7023964981707981
[LassoLarsIC] score :  0.6898359351037402
[LinearRegression] score :  0.7023964981707983
[LinearSVR] score :  0.5835294155952621
[MLPRegressor] score :  0.15077020254540252
[NuSVR] score :  0.5198856039303568
[OrthogonalMatchingPursuit] score :  0.5237691479421495  
[OrthogonalMatchingPursuitCV] score :  0.6512697618864467
[PLSRegression] score :  0.6678935858562085
[PassiveAggressiveRegressor] score :  0.6557643470337957 
[PoissonRegressor] score :  0.6005264504003165
[QuantileRegressor] score :  -0.007111100721483465
[RANSACRegressor] score :  -0.09800382677004826
[RadiusNeighborsRegressor] score :  0.3095224774306503   
[RandomForestRegressor] score :  0.713151208171273
[Ridge] score :  0.6979238255407658
[RidgeCV] score :  0.6979238255407707
[SGDRegressor] score :  0.6714958007573903
[SVR] score :  0.5590289037080705
[TheilSenRegressor] score :  0.6585949484336203
[TransformedTargetRegressor] score :  0.7023964981707983 
[TweedieRegressor] score :  0.17776316379923096
============================================================
[The Best score] :  0.8612048082268853
[The Best model] :  ExtraTreesRegressor
=========================================================
'''