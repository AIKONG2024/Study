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
r2 : [0.57451378 0.60918944 0.5857371  0.60672273 0.62440976] 
평균 r2 : 0.6001
eval r2 : // -4.967693863590719
        

=========================================================   
[AdaBoostRegressor]
r2 : [0.45631658 0.47842209 0.37567423 0.51468333 0.32980815] 
평균 r2 : 0.431
eval r2 : // 0.515523183772636
        

=========================================================   
[BaggingRegressor]
r2 : [0.77716375 0.77967358 0.77269421 0.77736869 0.79192066] 
평균 r2 : 0.7798
eval r2 : // 0.7392385365535447
        

=========================================================   
[BayesianRidge]
r2 : [0.57496416 0.60935215 0.58574179 0.60674961 0.62422209] 
평균 r2 : 0.6002
eval r2 : // -4.760713180080712
        

=========================================================   
[DecisionTreeRegressor]
r2 : [0.59659821 0.62155817 0.61899739 0.59260674 0.6161723 
] 
평균 r2 : 0.6092
eval r2 : // 0.523017648927697
        

=========================================================   
[DummyRegressor]
r2 : [-2.66476304e-04 -4.55521012e-05 -4.69694175e-05 -5.12572415e-04
 -4.65029679e-05]
평균 r2 : -0.0002
eval r2 : // -0.0007655020702366233


=========================================================   
[ElasticNet]
r2 : [-2.66476304e-04 -4.55521012e-05 -4.69694175e-05 -5.12572415e-04
 -4.65029679e-05]
평균 r2 : -0.0002
eval r2 : // -0.0007655020702366233


=========================================================   
[ElasticNetCV]
r2 : [0.58704914 0.60661997 0.57931171 0.59624531 0.61045919]
평균 r2 : 0.5959
eval r2 : // 0.6040447877582444


=========================================================   
[ExtraTreeRegressor]
r2 : [0.48809071 0.57083267 0.53819514 0.58846729 0.55658342]
평균 r2 : 0.5484
eval r2 : // 0.46299581519611865


=========================================================   
[GammaRegressor]
r2 : [0.01870869 0.01998681 0.01872012 0.01850521 0.01936086]
평균 r2 : 0.0191
eval r2 : // 0.01853815686462701

=========================================================   
[GaussianProcessRegressor]
r2 : [-5.27482966e+03 -2.44985826e+02 -3.61241286e+05 -1.64560873e-01
 -1.14114979e+01]
평균 r2 : -73354.5355
eval r2 : // -36106.3842611957


=========================================================   
[GradientBoostingRegressor]
r2 : [0.77437079 0.78586497 0.76874881 0.78967149 0.79812667]
평균 r2 : 0.7834
eval r2 : // 0.778396931459019


=========================================================   
[HistGradientBoostingRegressor]
r2 : [0.82377759 0.83447624 0.81770823 0.83812273 0.85061221]
평균 r2 : 0.8329
eval r2 : // 0.8100348715830935


=========================================================   
[HuberRegressor]
r2 : [0.53935087 0.60489774 0.54967305 0.59772568 0.61670396]
평균 r2 : 0.5817
eval r2 : // 0.5912319646072863


=========================================================   
[KNeighborsRegressor]
r2 : [0.69024087 0.70412976 0.67206266 0.69720717 0.71697993]
평균 r2 : 0.6961
eval r2 : // 0.6491476601757334


=========================================================   
[KernelRidge]
r2 : [0.51278016 0.54027126 0.51480062 0.52888437 0.53627688]
평균 r2 : 0.5266
eval r2 : // 0.5206991931201197


=========================================================   
[Lars]
r2 : [0.57438842 0.60915361 0.58543348 0.60684382 0.62448677]
평균 r2 : 0.6001
eval r2 : // -4.989876810424942


=========================================================   
[LarsCV]
r2 : [0.56011382 0.45644675 0.58530929 0.55650471 0.57774778]
평균 r2 : 0.5472
eval r2 : // -4.800801911402279


=========================================================   
[Lasso]
r2 : [-2.66476304e-04 -4.55521012e-05 -4.69694175e-05 -5.12572415e-04
 -4.65029679e-05]
평균 r2 : -0.0002
eval r2 : // -0.0007655020702366233


=========================================================   
[LassoCV]
r2 : [0.58414722 0.61064932 0.58419656 0.60294076 0.61719799]
평균 r2 : 0.5998
eval r2 : // -0.019123074210420254


=========================================================   
[LassoLars]
r2 : [-2.66476304e-04 -4.55521012e-05 -4.69694175e-05 -5.12572415e-04
 -4.65029679e-05]
평균 r2 : -0.0002
eval r2 : // -0.0007655020702366233


=========================================================   
[LassoLarsCV]
r2 : [0.56083306 0.45644675 0.58530929 0.55650471 0.57774778]
평균 r2 : 0.5474
eval r2 : // -5.106970617583157


=========================================================   
[LassoLarsIC]
r2 : [0.57438842 0.60915361 0.58543348 0.60684382 0.62448677]
평균 r2 : 0.6001
eval r2 : // -4.989699278706966


=========================================================   
[LinearRegression]
r2 : [0.57438842 0.60915361 0.58543348 0.60684382 0.62448677]
평균 r2 : 0.6001
eval r2 : // -4.989876810424952


=========================================================   
[LinearSVR]
r2 : [0.57092545 0.58810685 0.56691001 0.58032008 0.59334895]
평균 r2 : 0.5799
eval r2 : // 0.5808409964082957

=========================================================   
[MLPRegressor]
r2 : [0.66272681 0.68066955 0.66948909 0.69917527 0.72981469]
평균 r2 : 0.6884
eval r2 : // 0.6367733947638516


=========================================================   
[NuSVR]
r2 : [0.64948425 0.66801665 0.63975939 0.66018757 0.67820121]
평균 r2 : 0.6591
eval r2 : // 0.6483783079005576


=========================================================   
[OrthogonalMatchingPursuit]
r2 : [0.46706164 0.48897128 0.46178532 0.47826591 0.47542857]
평균 r2 : 0.4743
eval r2 : // 0.4680235525472034


=========================================================   
[OrthogonalMatchingPursuitCV]
r2 : [0.58073479 0.52102074 0.5759535  0.59280671 0.60784136]
평균 r2 : 0.5757
eval r2 : // -5.433155567238779


=========================================================   
[PLSRegression]
r2 : [0.51024522 0.53275865 0.51080338 0.52808615 0.53392619]
평균 r2 : 0.5232
eval r2 : // -5.903911684886296


=========================================================   
[PassiveAggressiveRegressor]
r2 : [-0.05718089  0.12028429 -0.98620283 -0.32354741  0.47193357]
평균 r2 : -0.1549
eval r2 : // 0.554188053317811


=========================================================   
[PoissonRegressor]
r2 : [0.04024115 0.04259222 0.04011906 0.04015569 0.0415647 
]
평균 r2 : 0.0409
'''