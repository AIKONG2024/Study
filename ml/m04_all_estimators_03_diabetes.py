from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.72, random_state=335688) 

# 모델구성
# allAlgorithms = all_estimators(type_filter='classifier') #41개
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
[ARDRegression] score :  0.6350312749564182
[AdaBoostRegressor] score :  0.5437700127782725
[BaggingRegressor] score :  0.5110314862795908
[BayesianRidge] score :  0.6311031840716759
[DecisionTreeRegressor] score :  -0.19765204814552972
[DummyRegressor] score :  -0.0035682709123607825
[ElasticNet] score :  0.0049585812120944706
[ElasticNetCV] score :  0.532043451051565
[ExtraTreeRegressor] score :  0.05738724594428268
[ExtraTreesRegressor] score :  0.556429157475046
[GammaRegressor] score :  0.0026889206340402483
[GaussianProcessRegressor] score :  -11.948041639087212
[GradientBoostingRegressor] score :  0.48999592517087787
[HistGradientBoostingRegressor] score :  0.4842725928452519
[HuberRegressor] score :  0.6505589783375773
[KNeighborsRegressor] score :  0.5276847973555605
[KernelRidge] score :  -2.9517699834325386
[Lars] score :  -1.6829656997991402
[LarsCV] score :  -1.184369918072838
[Lasso] score :  0.3730586047682336
[LassoCV] score :  0.6394589276980325
[LassoLars] score :  0.4280959054280915
[LassoLarsCV] score :  0.639212761485459
[LassoLarsIC] score :  0.6204633281189172
[LinearRegression] score :  0.6458792578583585
[LinearSVR] score :  -0.2805334914373401
[MLPRegressor] score :  -2.639627034029793
[NuSVR] score :  0.16168978378422139
[OrthogonalMatchingPursuit] score :  0.43293698067794684
[MLPRegressor] score :  -2.639627034029793
[NuSVR] score :  0.16168978378422139
[OrthogonalMatchingPursuit] score :  0.43293698067794684 
[OrthogonalMatchingPursuitCV] score :  0.6410893324680056

[PLSRegression] score :  0.6332509318493873
[PassiveAggressiveRegressor] score :  0.5833786583003528 
[PoissonRegressor] score :  0.3728783672838435
[QuantileRegressor] score :  -0.006655562002880311       
[RANSACRegressor] score :  0.3959746570875843
[RadiusNeighborsRegressor] score :  -0.0035682709123607825
[RandomForestRegressor] score :  0.51961954091913        
[Ridge] score :  0.48114590222627796
[RidgeCV] score :  0.6201301908546879
[SGDRegressor] score :  0.46731692419377235
[SVR] score :  0.17775539704588583
[TheilSenRegressor] score :  0.6417878175315133
[TransformedTargetRegressor] score :  0.6458792578583585 
[TweedieRegressor] score :  0.003079541416374121
====================================================================================================
[The Best score] :  0.6505589783375773
[The Best model] :  HuberRegressor
====================================================================================================
'''
