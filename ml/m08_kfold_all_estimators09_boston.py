import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
import warnings
warnings.filterwarnings('ignore')

datasets= load_boston()
x = datasets.data
y = datasets.target

#데이터 전처리
from sklearn.model_selection import train_test_split
#데이터 분류
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
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

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
=========================================================
[ARDRegression]
r2 : [0.6760271  0.77414259 0.7533657  0.66714905 0.77511233]
평균 r2 : 0.7292
eval r2 : // 0.5987267323636212


=========================================================
[AdaBoostRegressor]
r2 : [0.79668417 0.86897262 0.82703538 0.75188284 0.76349129]
평균 r2 : 0.8016
eval r2 : // 0.585976691968199


=========================================================
[BaggingRegressor]
r2 : [0.8398426  0.90609033 0.83222981 0.82584775 0.771287  ]
평균 r2 : 0.8351
eval r2 : // 0.6319454461683016


=========================================================
[BayesianRidge]
r2 : [0.68044924 0.77214307 0.75963129 0.66944718 0.7720165 ]
평균 r2 : 0.7307
eval r2 : // 0.6303560989153766


=========================================================
[DecisionTreeRegressor]
r2 : [0.80766413 0.87981627 0.54702081 0.6877792  0.68051044]
평균 r2 : 0.7206
eval r2 : // -0.01091390301943762


=========================================================
[DummyRegressor]
r2 : [-0.01848452 -0.0229174  -0.01542779 -0.02574903 -0.00155047]
평균 r2 : -0.0168
eval r2 : // -0.022822584374217136


=========================================================
[ElasticNet]
r2 : [0.11391596 0.16003684 0.14986138 0.12945334 0.21309425]
평균 r2 : 0.1533
eval r2 : // 0.10268740405239063


=========================================================
[ElasticNetCV]
r2 : [0.68232108 0.77075884 0.75863387 0.66881129 0.77227063]
평균 r2 : 0.7306
eval r2 : // 0.6308661230247256


=========================================================
[ExtraTreeRegressor]
r2 : [0.873706   0.85159956 0.59971789 0.59217599 0.54486267]
평균 r2 : 0.6924
eval r2 : // 0.40322764348915896


=========================================================
[ExtraTreesRegressor]
r2 : [0.8735963  0.9308597  0.91771564 0.81354152 0.84954601]
평균 r2 : 0.8771
eval r2 : // 0.7520350124351928


=========================================================
[GammaRegressor]
r2 : [0.14764525 0.19823657 0.18190884 0.16956126 0.23517268]
평균 r2 : 0.1865
eval r2 : // 0.14482363604179682


=========================================================
[GaussianProcessRegressor]
평균 r2 : -0.2111
eval r2 : // 0.4375214891006459


=========================================================
[GradientBoostingRegressor]
r2 : [0.87156135 0.93994572 0.88055488 0.84326501 0.8434726 ]
평균 r2 : 0.8758
eval r2 : // 0.6820868223264668


=========================================================
[HistGradientBoostingRegressor]
r2 : [0.83394707 0.9317798  0.79475614 0.82166151 0.83940767]
평균 r2 : 0.8443
eval r2 : // 0.5765427486080261


=========================================================     
[HuberRegressor]
r2 : [0.63434042 0.77293051 0.77679702 0.64549624 0.76276353] 
평균 r2 : 0.7185
eval r2 : // 0.5983620682902944
        

=========================================================     
[KNeighborsRegressor]
r2 : [0.69801547 0.75691911 0.69315584 0.56973607 0.76411557] 
평균 r2 : 0.6964
eval r2 : // 0.451558361063906


=========================================================
[KernelRidge]
r2 : [0.61418138 0.67863894 0.69407435 0.63519342 0.69000398]
평균 r2 : 0.6624
eval r2 : // 0.5575112393813774


=========================================================
[Lars]
r2 : [0.6785371  0.77203711 0.75917131 0.66871851 0.77092775]
평균 r2 : 0.7299
eval r2 : // 0.5992559113894006


=========================================================
[LarsCV]
r2 : [0.6785371  0.7695672  0.76186512 0.66981269 0.77308799]
평균 r2 : 0.7306
eval r2 : // 0.5766971792145985


=========================================================
[Lasso]
r2 : [0.17849107 0.22568961 0.25423141 0.23896471 0.34538322]
평균 r2 : 0.2486
eval r2 : // 0.15026525142975355


=========================================================
[LassoCV]
r2 : [0.67854407 0.77340991 0.75732701 0.67000974 0.77178908]
평균 r2 : 0.7302
eval r2 : // 0.6163212089225044


=========================================================
[LassoLars]
r2 : [-0.01848452 -0.0229174  -0.01542779 -0.02574903 -0.00155047]
평균 r2 : -0.0168
eval r2 : // -0.022822584374217136


=========================================================
[LassoLarsCV]
r2 : [0.6785371  0.77309511 0.76298499 0.66981269 0.77352016]
평균 r2 : 0.7316
eval r2 : // 0.5907904842524481


=========================================================
[LassoLarsIC]
r2 : [0.6785371  0.77318489 0.76249994 0.66871316 0.77298523]
평균 r2 : 0.7312
eval r2 : // 0.6119717843397575


=========================================================
[LinearRegression]
r2 : [0.6785371  0.77203711 0.75917131 0.66871851 0.77092775]
평균 r2 : 0.7299
eval r2 : // 0.5992559113894003


=========================================================
[LinearSVR]
r2 : [0.5388338  0.64795823 0.64377695 0.57676181 0.67188214]
평균 r2 : 0.6158
eval r2 : // 0.35349973875659035


=========================================================
[MLPRegressor]
r2 : [0.15726659 0.08689194 0.13639298 0.18173    0.02952817]
평균 r2 : 0.1184
eval r2 : // -0.5674785274455403


=========================================================
[NuSVR]
r2 : [0.46128305 0.64795089 0.56960807 0.42251212 0.58984187]
평균 r2 : 0.5382
eval r2 : // 0.28960147261505875


=========================================================
[OrthogonalMatchingPursuit]
r2 : [0.4980532  0.44452792 0.4971378  0.52906267 0.5409354 ]
평균 r2 : 0.5019
eval r2 : // 0.4751627038962345


=========================================================
[OrthogonalMatchingPursuitCV]
r2 : [0.64437837 0.74509961 0.7380436  0.64305714 0.75212195]
평균 r2 : 0.7045
eval r2 : // 0.5091601029001407


=========================================================
[PLSRegression]
r2 : [0.65097783 0.7420086  0.78490242 0.6150285  0.7591255 ]
평균 r2 : 0.7104
eval r2 : // 0.6205134180169998


=========================================================
[PassiveAggressiveRegressor]
r2 : [0.46325994 0.60783827 0.60206127 0.6272035  0.71451944]
평균 r2 : 0.603
eval r2 : // 0.5387292567781637


=========================================================
[PoissonRegressor]
r2 : [0.57643007 0.67672752 0.63680035 0.58553939 0.66010538]
평균 r2 : 0.6271
eval r2 : // 0.56081450636313


=========================================================
[QuantileRegressor]
r2 : [-0.05790354 -0.00173986 -0.00012607 -0.09868283 -0.02321748]
평균 r2 : -0.0363
eval r2 : // -0.06374080302434759


=========================================================
[RANSACRegressor]
r2 : [0.58365857 0.56358847 0.64599379 0.42932428 0.69070144]
평균 r2 : 0.5827
eval r2 : // 0.299293460623964


=========================================================
[RadiusNeighborsRegressor]
r2 : [0.2543848  0.42044491 0.30249317 0.23088738 0.35585689]
평균 r2 : 0.3128
eval r2 : // 0.40393333352608607


=========================================================
[RandomForestRegressor]
r2 : [0.8510654  0.9393946  0.85525851 0.82838447 0.80528079]
평균 r2 : 0.8559
eval r2 : // 0.6484477052538689


=========================================================
[Ridge]
r2 : [0.68262089 0.76813493 0.75552822 0.66747209 0.76933524]
평균 r2 : 0.7286
eval r2 : // 0.6320829009147949


=========================================================
[RidgeCV]
r2 : [0.67978926 0.77218074 0.75951312 0.66919082 0.77157706]
평균 r2 : 0.7305
eval r2 : // 0.6052483430721269


=========================================================
[SGDRegressor]
r2 : [0.67036103 0.74633908 0.7566389  0.66413018 0.7602962 ]
평균 r2 : 0.7196
eval r2 : // 0.6227123904917469


=========================================================
[SVR]
r2 : [0.46815218 0.65876577 0.60533429 0.46415208 0.62907094]
평균 r2 : 0.5651
eval r2 : // 0.28885785646049367


=========================================================
[TheilSenRegressor]
r2 : [0.61369626 0.73326855 0.77655075 0.62917577 0.78272464]
평균 r2 : 0.7071
eval r2 : // 0.607692612632712


=========================================================
[TransformedTargetRegressor]
r2 : [0.6785371  0.77203711 0.75917131 0.66871851 0.77092775]
평균 r2 : 0.7299
eval r2 : // 0.5992559113894003


=========================================================
[TweedieRegressor]
r2 : [0.13113453 0.20130317 0.16624105 0.15354524 0.23090247]
평균 r2 : 0.1766
eval r2 : // 0.14507327919043012

============================================================
[The Best score] :  0.7520350124351928
[The Best model] :  ExtraTreesRegressor
============================================================
'''