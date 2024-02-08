# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null

x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
y = train_csv['count'] 

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8
)

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
# allAlgorithms = all_estimators(type_filter='classifier') #41개
allAlgorithms = all_estimators(type_filter='regressor') #55개
best_score = 0
best_model = ""

# 모델구성
from sklearn.metrics import r2_score
for name, algorithm in allAlgorithms :
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kf)
        y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
        r2 = r2_score(y_test, y_predict)
        # 모델
        if best_score < r2:
            best_score = r2
            best_model = name
        # 3. 훈련
        # 평가, 예측
        print(f"""
=========================================================
[{name}]
r2 : {scores} 
평균 r2 : {round(np.mean(scores),4)}
eval r2 : // {r2}
        """)
    except:
        continue
    
print("="*60)
print("[The Best score] : ", best_score )
print("[The Best model] : ", best_model )
print("="*60)

'''
[ARDRegression] score :  0.5845659116942068
[AdaBoostRegressor] score :  0.6118955756327934
[BaggingRegressor] score :  0.7451483287504497
[BayesianRidge] score :  0.586389276493887
[DecisionTreeRegressor] score :  0.6211025860947623
[DummyRegressor] score :  -0.0008204650381971046
[ElasticNet] score :  0.57379084183296
[ElasticNetCV] score :  0.5885433872236381
[ExtraTreeRegressor] score :  0.5021380255635675
[ExtraTreesRegressor] score :  0.7831273874121932
[GammaRegressor] score :  0.42857345376533507
[GaussianProcessRegressor] score :  0.5438484715691424
[GradientBoostingRegressor] score :  0.7523202736398389
[HistGradientBoostingRegressor] score :  0.7850846854458855
[HuberRegressor] score :  0.5758373537299705
[KNeighborsRegressor] score :  0.6701105116314219
[KernelRidge] score :  -1.1177648924486943
[Lars] score :  0.5856523615621383
[LarsCV] score :  0.5857569854966341
[Lasso] score :  0.5854655555930579
[LassoCV] score :  0.5858187617472128
[LassoLars] score :  0.33593184190122805
[LassoLarsCV] score :  0.5857569854966341
[LassoLarsIC] score :  0.5857261666321981
[LinearRegression] score :  0.5856523615621383
[LinearSVR] score :  0.5353099415855508
[MLPRegressor] score :  0.5829014362330467
[NuSVR] score :  0.44429504964325495
[OrthogonalMatchingPursuit] score :  0.368180097404306
[OrthogonalMatchingPursuitCV] score :  0.5598913412793817
[PLSRegression] score :  0.5912719949696983
[PassiveAggressiveRegressor] score :  0.5237414015941809
[PoissonRegressor] score :  0.6130410580890597
[QuantileRegressor] score :  -0.01404246873285131
[RANSACRegressor] score :  0.5304011312156287
[RandomForestRegressor] score :  0.7616196172144681
[Ridge] score :  0.5857252288645509
[RidgeCV] score :  0.586345776243441
[SGDRegressor] score :  0.585956469847675
[SVR] score :  0.43752209216232907
[TheilSenRegressor] score :  0.5660955601108255
[TransformedTargetRegressor] score :  0.5856523615621383
[TweedieRegressor] score :  0.5418418472661239
============================================================
[The Best score] :  0.7850846854458855
[The Best model] :  HistGradientBoostingRegressor        
========================================================
dacon
[ExtraTreesRegressor] score :  0.7885106816400261
mse loss : 1473.2397773972602
46.35832
'''