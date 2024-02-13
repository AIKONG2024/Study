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
=========================================================
[ARDRegression]
r2 : [0.57985555 0.64070055 0.60633846 0.59638342 0.50778089]
평균 r2 : 0.5862
eval r2 : // 0.5757967561635


=========================================================
[AdaBoostRegressor]
r2 : [0.58604017 0.59989586 0.66617195 0.51480898 0.51184622]
평균 r2 : 0.5758
eval r2 : // 0.6230733168204097


=========================================================
[BaggingRegressor]
r2 : [0.70193953 0.79710835 0.77831521 0.73324789 0.69727681]
평균 r2 : 0.7416
eval r2 : // 0.6888385694751267


=========================================================
[BayesianRidge]
r2 : [0.58045061 0.63894021 0.60658176 0.59817382 0.50592781]
평균 r2 : 0.586
eval r2 : // 0.5847342772243711


=========================================================
[DecisionTreeRegressor]
r2 : [0.55045074 0.60503399 0.60227675 0.64197051 0.4672878 ]
평균 r2 : 0.5734
eval r2 : // 0.49801137656470706


=========================================================
[DummyRegressor]
r2 : [-7.78973010e-03 -4.32609778e-03 -5.04793535e-03 -1.41769784e-02
 -1.68204208e-07]
평균 r2 : -0.0063
eval r2 : // -0.011180767338639708


=========================================================
[ElasticNet]
r2 : [0.19607139 0.23628868 0.19866839 0.21862171 0.20702632]
평균 r2 : 0.2113
eval r2 : // 0.20012649292314877


=========================================================
[ElasticNetCV]
r2 : [0.56739801 0.62597349 0.58228598 0.58331989 0.50220036]
평균 r2 : 0.5722
eval r2 : // 0.5781960177562289


=========================================================
[ExtraTreeRegressor]
r2 : [0.57645407 0.53741492 0.56373855 0.52792697 0.56070035]
평균 r2 : 0.5532
eval r2 : // 0.4716564256100423


=========================================================
[ExtraTreesRegressor]
r2 : [0.78168167 0.84433325 0.80790782 0.77329372 0.72816401]
평균 r2 : 0.7871
eval r2 : // 0.7071647285956232


=========================================================
[GammaRegressor]
r2 : [0.09603676 0.10861853 0.09208291 0.08959359 0.09302623]
평균 r2 : 0.0959
eval r2 : // 0.11649048066825363


=========================================================
[GaussianProcessRegressor]
r2 : [-23.60590734 -31.19804007 -54.05523584 -37.54919393 -25.06436472]
평균 r2 : -34.2945
eval r2 : // -6.884959149030116


=========================================================
[GradientBoostingRegressor]
r2 : [0.74442706 0.81879307 0.78141006 0.73118911 0.71332631]
평균 r2 : 0.7578
eval r2 : // 0.6882538623652154


=========================================================
[HistGradientBoostingRegressor]
r2 : [0.77996753 0.81267064 0.79004224 0.73612414 0.75770794]
평균 r2 : 0.7753
eval r2 : // 0.6859391469830427


=========================================================
[HuberRegressor]
r2 : [0.55459517 0.6421919  0.58457008 0.59609642 0.51129564]
평균 r2 : 0.5777
eval r2 : // 0.5760497643310902


=========================================================
[KNeighborsRegressor]
r2 : [0.72107542 0.7491868  0.71787005 0.74487944 0.61923085]
평균 r2 : 0.7104
eval r2 : // 0.5246582239572162


=========================================================
[KernelRidge]
r2 : [0.58009146 0.63858832 0.60571741 0.59938484 0.50722781]
평균 r2 : 0.5862
eval r2 : // 0.5876982341525583


=========================================================
[Lars]
r2 : [0.58018571 0.63892268 0.60697576 0.59825339 0.50555632]
평균 r2 : 0.586
eval r2 : // 0.5824932979800316


=========================================================
[LarsCV]
r2 : [0.58010103 0.64072035 0.60637849 0.59638831 0.50630171]
평균 r2 : 0.586
eval r2 : // 0.5867571024643379


=========================================================
[Lasso]
r2 : [0.56438069 0.63629805 0.58335906 0.54192082 0.49147405]
평균 r2 : 0.5635
eval r2 : // 0.5683093007437718


=========================================================
[LassoCV]
r2 : [0.58030509 0.6408149  0.60656399 0.59623044 0.50574952]
평균 r2 : 0.5859
eval r2 : // 0.5836027993596871


=========================================================
[LassoLars]
r2 : [0.32253186 0.38916644 0.33118946 0.35631342 0.33446033]
평균 r2 : 0.3467
eval r2 : // 0.5031879764227023


=========================================================
[LassoLarsCV]
r2 : [0.57996382 0.64066563 0.60574018 0.59509886 0.50570405]
평균 r2 : 0.5854
eval r2 : // 0.5840621448607055


=========================================================
[LassoLarsIC]
r2 : [0.5801188  0.63998027 0.60645908 0.5980666  0.50555632]
평균 r2 : 0.586
eval r2 : // 0.5856366621692812


=========================================================
[LinearRegression]
r2 : [0.58018571 0.63892268 0.60697576 0.59825339 0.50555632]
평균 r2 : 0.586
eval r2 : // 0.5824932979800312


=========================================================
[LinearSVR]
r2 : [0.43203731 0.53586025 0.44378442 0.49457012 0.42836799]
평균 r2 : 0.4669
eval r2 : // 0.30539394401617614


=========================================================
[MLPRegressor]
r2 : [0.47642789 0.49345317 0.42520219 0.47704619 0.41810186]
평균 r2 : 0.458
eval r2 : // -0.06551741048416226


=========================================================
[NuSVR]
r2 : [0.4329027  0.52668443 0.45312746 0.51562813 0.43533054]
평균 r2 : 0.4727
eval r2 : // 0.19254132973037752


=========================================================
[OrthogonalMatchingPursuit]
r2 : [0.36055277 0.42804689 0.38864635 0.28585966 0.26902272]
평균 r2 : 0.3464
eval r2 : // 0.36428537835494745


=========================================================
[OrthogonalMatchingPursuitCV]
r2 : [0.57460161 0.64041312 0.59406356 0.57166656 0.48870884]
평균 r2 : 0.5739
eval r2 : // 0.5659427377117219


=========================================================
[PLSRegression]
r2 : [0.57528982 0.62302819 0.59911865 0.59819215 0.50111994]
평균 r2 : 0.5793
eval r2 : // 0.5822033144488914


=========================================================
[PassiveAggressiveRegressor]
r2 : [0.54357881 0.62349353 0.52878901 0.56380373 0.46465698]
평균 r2 : 0.5449
eval r2 : // 0.5576286237341412


=========================================================
[PoissonRegressor]
r2 : [0.60801827 0.65381009 0.61864644 0.61039879 0.5238641 ]
평균 r2 : 0.6029
eval r2 : // 0.6257001567578805


=========================================================
[QuantileRegressor]
r2 : [-0.05083144 -0.00945329 -0.05139678 -0.00407582 -0.03463576]
평균 r2 : -0.0301
eval r2 : // -0.019098637417257258


=========================================================
[RANSACRegressor]
r2 : [0.43349804 0.60366994 0.41719071 0.44088078 0.3499103 ]
평균 r2 : 0.449
eval r2 : // 0.528700086975566


=========================================================
[RadiusNeighborsRegressor]
r2 : [0.29905747 0.33085767 0.28333426 0.33806983 0.28352535]
평균 r2 : 0.307
eval r2 : // 0.30321248941425205


=========================================================
[RandomForestRegressor]
r2 : [0.72680404 0.82067587 0.78343145 0.74053232 0.7274702 ]
평균 r2 : 0.7598
eval r2 : // 0.6920561968719705


=========================================================
[Ridge]
r2 : [0.58056493 0.6388705  0.60617511 0.59804862 0.50617315]
평균 r2 : 0.586
eval r2 : // 0.5858550223164762


=========================================================
[RidgeCV]
r2 : [0.58056493 0.6388705  0.60617511 0.59804862 0.50617315]
평균 r2 : 0.586
eval r2 : // 0.5858550223164739


=========================================================
[SGDRegressor]
r2 : [0.57911382 0.63439324 0.60447108 0.59639312 0.5027498 ]
평균 r2 : 0.5834
eval r2 : // 0.5842504124014353


=========================================================
[SVR]
r2 : [0.42206976 0.53412658 0.44484632 0.51720394 0.42911596]
평균 r2 : 0.4695
eval r2 : // 0.2635292808363279


=========================================================
[TheilSenRegressor]
r2 : [0.58748671 0.63100856 0.59661195 0.54849243 0.48837471]
평균 r2 : 0.5704
eval r2 : // 0.5721803916896727


=========================================================
[TransformedTargetRegressor]
r2 : [0.58018571 0.63892268 0.60697576 0.59825339 0.50555632]
평균 r2 : 0.586
eval r2 : // 0.5824932979800312


=========================================================
[TweedieRegressor]
r2 : [0.11946876 0.14646271 0.12214111 0.13253279 0.13118341]
평균 r2 : 0.1304
eval r2 : // 0.12027881867439583

============================================================
[The Best score] :  0.7071647285956232
[The Best model] :  ExtraTreesRegressor
============================================================
'''