import pandas as pd
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

path = 'C:/_data/kaggle/bike/'
train_csv =pd.read_csv(path + 'train.csv', index_col=0)
test_csv =pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

#데이터 전처리
x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
y = train_csv['count']
    
#데이터
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, random_state= 12345)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

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
        x_predict = model.predict(x_test)
        mse = mean_squared_error(y_test, x_predict)
        if best_acc < results:
            best_acc = results
            best_model = name
        print(f"[{name}] score : ", results)
        print("mse loss :", mse)

        y_submit = model.predict(test_csv) # count 값이 예측됨.
        submission_csv['count'] = y_submit

        ######### submission.csv 만들기(count컬럼에 값만 넣어주면됨) ############
        import time as tm
        ltm = tm.localtime(tm.time())
        save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}_{mse}_" 
        file_path = path + f"submission_{save_time}.csv"
        submission_csv.to_csv(file_path, index=False)

    except:
        continue
print("="*60)
print("[The Best score] : ", best_acc )
print("[The Best model] : ", best_model )
print("="*60)
'''
[ARDRegression] score :  0.2581703301470566
[AdaBoostRegressor] score :  0.209528219867688
[BaggingRegressor] score :  0.2097057100197809
[BayesianRidge] score :  0.25751479389647114
[DecisionTreeRegressor] score :  -0.2450722983083391
[DummyRegressor] score :  -7.99508081730238e-05
[ElasticNet] score :  0.24047067322843052
[ElasticNetCV] score :  0.2571194225053902
[ExtraTreeRegressor] score :  -0.1968314551154935
[ExtraTreesRegressor] score :  0.15525024601621773
[GammaRegressor] score :  0.14691796556115377
[GammaRegressor] score :  0.14691796556115377
[GaussianProcessRegressor] score :  -25511.648955220397
[GradientBoostingRegressor] score :  0.32823114494105154
[HistGradientBoostingRegressor] score :  0.35691020073346846
[HuberRegressor] score :  0.2386419384485854
[KNeighborsRegressor] score :  0.3063187160588716
[KernelRidge] score :  -0.849060541830063
[Lars] score :  0.2574230473244603
[LarsCV] score :  0.2582739633257378
[Lasso] score :  0.25836151907645855
[LassoCV] score :  0.25830307758983906
[LassoLars] score :  -7.99508081730238e-05
[LassoLarsCV] score :  0.25815787837313686
[LassoLarsIC] score :  0.25828408222904564
[LinearRegression] score :  0.2574230473244602
[LinearSVR] score :  0.21971307739449364
[MLPRegressor] score :  0.30266505942415634
[NuSVR] score :  0.22721101746286887
[OrthogonalMatchingPursuit] score :  0.161502446200123
[OrthogonalMatchingPursuitCV] score :  0.25642367599125104
[PLSRegression] score :  0.2507061187636479
[PassiveAggressiveRegressor] score :  0.21618611307010094[PoissonRegressor] score :  0.26434194370148645
[QuantileRegressor] score :  -0.06112347949410846
[RANSACRegressor] score :  0.008668971026946415
[RadiusNeighborsRegressor] score :  -2.0951894430051372e+31
[RandomForestRegressor] score :  0.2665969498124511
[Ridge] score :  0.25742776003161616
[RidgeCV] score :  0.2574641354217828
[SGDRegressor] score :  0.25704926787030935
[SVR] score :  0.21959712977977397
[TheilSenRegressor] score :  0.23716287148505522
[TransformedTargetRegressor] score :  0.2574230473244602
[TweedieRegressor] score :  0.21805289119985505
============================================================
[The Best score] :  0.35691020073346846
[The Best model] :  HistGradientBoostingRegressor
============================================================
'''