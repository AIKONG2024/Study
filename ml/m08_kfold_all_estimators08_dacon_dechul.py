import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

path = 'C:/_data/dacon/dechul/'
#데이터 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

unique, count = np.unique(train_csv['근로기간'], return_counts=True)
unique, count = np.unique(test_csv['근로기간'], return_counts=True)
train_le = LabelEncoder()
test_le = LabelEncoder()
train_csv['주택소유상태'] = train_le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = train_le.fit_transform(train_csv['대출목적'])
train_csv['근로기간'] = train_le.fit_transform(train_csv['근로기간'])
train_csv['대출등급'] = train_le.fit_transform(train_csv['대출등급'])


test_csv['주택소유상태'] = test_le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = test_le.fit_transform(test_csv['대출목적'])
test_csv['근로기간'] = test_le.fit_transform(test_csv['근로기간'])

#3. split 수치화 대상 int로 변경: 대출기간
train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(float)
test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(float)

x = train_csv.drop('대출등급', axis=1)
y = train_csv['대출등급']

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
from sklearn.metrics import accuracy_score
for name, algorithm in allAlgorithms :
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kf)
        y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
        acc_score = accuracy_score(y_test, y_predict)
        # 모델
        if best_score < acc_score:
            best_score = acc_score
            best_model = name
        # 3. 훈련
        # 평가, 예측
        print(f"""
=========================================================
[{name}]
acc : {scores} 
평균 acc : {round(np.mean(scores),4)}
eval acc : // {acc_score}
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
eval r2 : // 0.575796756163437
        

=========================================================   
[AdaBoostRegressor]
r2 : [0.59212085 0.48920327 0.63445939 0.4985253  0.5704539 
] 
평균 r2 : 0.557
eval r2 : // 0.607539052048256
        

=========================================================   
[BaggingRegressor]
r2 : [0.70666794 0.78669612 0.78017678 0.72361727 0.70459518] 
평균 r2 : 0.7404
eval r2 : // 0.6821768174430185
        

=========================================================   
[BayesianRidge]
r2 : [0.58045061 0.63894021 0.60658176 0.59817382 0.50592781] 
평균 r2 : 0.586
eval r2 : // 0.5847342772243711
        

=========================================================   
[DecisionTreeRegressor]
r2 : [0.5563206  0.57190922 0.58691187 0.5938457  0.51904767] 
평균 r2 : 0.5656
eval r2 : // 0.4793849532663841
        

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
eval r2 : // 0.578196017756229


=========================================================   
[ExtraTreeRegressor]
r2 : [0.47408233 0.62552698 0.58387392 0.60299527 0.41164721]
평균 r2 : 0.5396
eval r2 : // 0.36631239729837517


=========================================================   
[ExtraTreesRegressor]
r2 : [0.78435775 0.8463908  0.79615108 0.77434902 0.74123628]
평균 r2 : 0.7885
eval r2 : // 0.7084785555071434


=========================================================   
[GammaRegressor]
r2 : [0.09603676 0.10861853 0.09208291 0.08959359 0.09302623]
평균 r2 : 0.0959
eval r2 : // 0.11649048066825363


=========================================================   
[GaussianProcessRegressor]
r2 : [-23.60590776 -31.19803973 -54.0552374  -37.54919464 -25.06436433]
평균 r2 : -34.2945
eval r2 : // -6.884959149020134


=========================================================   
[GradientBoostingRegressor]
r2 : [0.74407814 0.81803934 0.78288616 0.73104645 0.71256583]
평균 r2 : 0.7577
eval r2 : // 0.6874792818996666


=========================================================   
[HistGradientBoostingRegressor]
r2 : [0.77996753 0.81267064 0.79004224 0.73612414 0.75770794]
평균 r2 : 0.7753
eval r2 : // 0.6859391469830427


=========================================================   
[HuberRegressor]
r2 : [0.55459517 0.6421919  0.58457008 0.59609642 0.51129564]
평균 r2 : 0.5777
eval r2 : // 0.5760497643307568


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
eval r2 : // 0.5824932979800311


=========================================================   
[LarsCV]
r2 : [0.58010103 0.64072035 0.60637849 0.59638831 0.50630171]
평균 r2 : 0.586
eval r2 : // 0.5867571024643377


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
eval r2 : // 0.5856366621692809


=========================================================   
[LinearRegression]
r2 : [0.58018571 0.63892268 0.60697576 0.59825339 0.50555632]
평균 r2 : 0.586
eval r2 : // 0.5824932979800312


=========================================================   
[LinearSVR]
r2 : [0.43609945 0.53331063 0.44131634 0.49687072 0.43049857]
평균 r2 : 0.4676
eval r2 : // 0.3055776135440855


=========================================================   
[MLPRegressor]
r2 : [0.44453914 0.50745876 0.42849957 0.42960673 0.42738576]
평균 r2 : 0.4475
eval r2 : // 0.004275247422880879


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
eval r2 : // 0.5659427377117215


=========================================================   
[PLSRegression]
r2 : [0.57528982 0.62302819 0.59911865 0.59819215 0.50111994]
평균 r2 : 0.5793
eval r2 : // 0.5822033144488914


=========================================================   
[PassiveAggressiveRegressor]
r2 : [0.57481201 0.55808424 0.58761248 0.58597369 0.4795966 
]
평균 r2 : 0.5572
eval r2 : // 0.5761213435831979


=========================================================   
[PoissonRegressor]
r2 : [0.60801827 0.65381009 0.61864644 0.61039879 0.5238641 
]
평균 r2 : 0.6029
eval r2 : // 0.6257001567577634


=========================================================   
[QuantileRegressor]
r2 : [-0.05083144 -0.00945329 -0.05139678 -0.00407582 -0.03463576]
평균 r2 : -0.0301
eval r2 : // -0.019098637417471753


=========================================================   
[RANSACRegressor]
r2 : [0.47800229 0.52835213 0.56851481 0.40657341 0.49320003]
평균 r2 : 0.4949
eval r2 : // 0.5236909723834857


=========================================================   
[RadiusNeighborsRegressor]
r2 : [0.29905747 0.33085767 0.28333426 0.33806983 0.28352535]
평균 r2 : 0.307
eval r2 : // 0.30321248941425205


=========================================================   
[RandomForestRegressor]
r2 : [0.7312229  0.81020957 0.78217344 0.74205557 0.72615729]
평균 r2 : 0.7584
eval r2 : // 0.6926055742345536


=========================================================   
[Ridge]
r2 : [0.58056493 0.6388705  0.60617511 0.59804862 0.50617315]
평균 r2 : 0.586
eval r2 : // 0.5858550223164762


=========================================================   
[RidgeCV]
r2 : [0.58056493 0.6388705  0.60617511 0.59804862 0.50617315]
평균 r2 : 0.586
eval r2 : // 0.5858550223164741


=========================================================   
[SGDRegressor]
r2 : [0.58070781 0.63561154 0.60373469 0.59716955 0.50080304]
평균 r2 : 0.5836
eval r2 : // 0.5848164614187492


=========================================================   
[SVR]
r2 : [0.42206976 0.53412658 0.44484632 0.51720394 0.42911596]
평균 r2 : 0.4695
eval r2 : // 0.2635292808363279


=========================================================   
[TheilSenRegressor]
r2 : [0.58753536 0.63127534 0.59515935 0.55533467 0.48859611]
평균 r2 : 0.5716
eval r2 : // 0.5714048588296443


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

============================================================[The Best score] :  0.7084785555071434
[The Best model] :  ExtraTreesRegressor
============================================================
'''