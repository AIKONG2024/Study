import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# 1. 데이터
load_cancer = load_breast_cancer()
x = load_cancer.data
y = load_cancer.target

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias= False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.001, 1),
    'max_depth' : hp.quniform('max_depth',3, 10, 1),
    'num_leaves' : hp.quniform('num_leaves', 24, 40,1),
    'min_child_samples' : hp.quniform('min_child_samples',10, 200, 1),
    'min_child_weight' : hp.quniform('min_child_weight',1,50, 1),
    'subsample' : hp.uniform('subsample',0.5, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    'min_bin' :  hp.quniform('min_bin', 9, 500, 1),
    'reg_lambda' : hp.uniform('reg_lambda',-0.001, 10),
    'reg_alpha' : hp.uniform('reg_alpha',0.01, 50)
}

def objective(search_space):
    params = {
        'n_estimators' : 100,
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']),
        'num_leaves': int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight': int(search_space['min_child_weight']),
        'subsample' : max(min(search_space['subsample'], 1), 0),
        'colsample_bytree' : search_space['colsample_bytree'],
        'min_bin' : max(int(search_space['min_bin']), 10),
        'reg_lambda':max(search_space['reg_lambda'], 0),
        'reg_alpha' : search_space['reg_alpha'],
    }
    
    model = XGBClassifier(**params, n_jobs = -1)
    
    model.fit(x_train, y_train,
              eval_set = [(x_train, y_train), (x_test,y_test)],
              eval_metric = 'logloss',
              verbose=0,
              early_stopping_rounds = 50,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    return results

trial_val = Trials()

start_time = time.time()
best = fmin(
    fn = objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals = 50,
    trials = trial_val,
    rstate= np.random.default_rng(seed=10)
)

end_time = time.time()
print("best", best)
import pandas as pd
losses = [dict.get('loss') for dict in trial_val.results]
df = pd.DataFrame( {
    'target' : losses,
    'max_depth' : trial_val.vals['max_depth'],
    'num_leaves' : trial_val.vals['num_leaves'],
    'min_child_samples' : trial_val.vals['min_child_samples'],
    'min_child_weight' : trial_val.vals['min_child_weight'],
    'subsample' : trial_val.vals['subsample'],
    'colsample_bytree' : trial_val.vals['colsample_bytree'],
    'min_bin' : trial_val.vals['min_bin'],
    'reg_lambda' : trial_val.vals['reg_lambda'],
    'reg_alpha' : trial_val.vals['reg_alpha']})
print(df)

print("tunning time : ", round(end_time - start_time, 2))

'''
best {'colsample_bytree': 0.7339435697749975, 'learning_rate': 0.7347171995853888, 'max_depth': 8.0, 'min_bin': 42.0, 'min_child_samples': 26.0, 'min_child_weight': 48.0, 'num_leaves': 29.0, 'reg_alpha': 36.13373729410798, 'reg_lambda': 8.400175844097886, 'subsample': 0.675647675869977}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.956140        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.964912        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.631579        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.947368        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.991228        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.973684        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.947368        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.964912       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.956140        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.956140        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.956140       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.947368        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.956140        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.912281        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.973684        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.947368       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.938596        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.973684        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.631579        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.956140        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.631579        9.0        26.0               43.0              49.0   0.705218          0.646436    488.0    6.109137  21.424206
21  0.631579        9.0        25.0               46.0              49.0   0.708567          0.637370    497.0    5.892463  21.259449
22  0.631579        9.0        25.0               37.0              40.0   0.528146          0.666236     15.0    8.417731  36.404716
23  0.631579        9.0        24.0               34.0              41.0   0.539073          0.643010    472.0    7.852726  10.776741
24  0.631579        9.0        25.0               62.0              43.0   0.574763          0.798693    496.0    3.059584  11.204504
25  0.631579       10.0        24.0               69.0              40.0   0.524651          0.906346    444.0    0.092793  15.246895
26  0.631579       10.0        24.0               73.0              39.0   0.503924          0.924927    451.0    0.404915  11.479262
27  0.631579       10.0        27.0              130.0              37.0   0.594562          0.933644    339.0    2.945779   9.641429
28  0.631579       10.0        27.0              128.0              35.0   0.507749          0.905389    348.0    0.399882  16.347230
29  0.631579       10.0        27.0              129.0              36.0   0.584264          0.901828    350.0    3.118343  18.015438
30  0.631579        6.0        27.0              195.0              44.0   0.500245          0.885899    349.0    2.436770  26.420479
31  0.631579        6.0        31.0              197.0              46.0   0.560703          0.861979    146.0    2.244394  27.306610
32  0.631579        6.0        31.0               95.0              46.0   0.714844          0.991727    128.0    3.816849  18.936513
33  0.973684        6.0        30.0               97.0              46.0   0.990057          0.986507     93.0    3.944375  27.392303
34  0.956140        7.0        35.0               77.0              50.0   0.952233          0.598613    173.0    5.911392   7.798387
35  0.631579        8.0        25.0               29.0              39.0   0.623482          0.780448     18.0    9.423861  36.843307
36  0.956140        8.0        40.0               26.0              33.0   0.630870          0.681020     15.0    9.132641  37.563302
37  0.964912        7.0        39.0               59.0               7.0   0.792009          0.502443    250.0    6.105163  22.923200
38  0.631579        5.0        38.0               47.0              50.0   0.784217          0.597199    390.0    6.542369  20.509527
39  0.947368        3.0        38.0               83.0              20.0   0.876021          0.584910    102.0    8.021716  45.543006
40  0.631579        9.0        29.0              136.0              42.0   0.535727          0.684355    469.0    7.431274   5.560704
41  0.631579        4.0        38.0               17.0              48.0   0.704387          0.523098    388.0    5.380945  20.434771
42  0.973684        8.0        26.0               42.0              11.0   0.693254          0.760234    415.0    6.842205  12.901137
43  0.964912        9.0        28.0               85.0              23.0   0.725128          0.743047    439.0    4.644291  15.487088
44  0.956140       10.0        28.0              103.0              38.0   0.812302          0.949328    316.0    0.005078  29.042011
45  0.947368        8.0        24.0               10.0              27.0   0.668269          0.814820    224.0    8.855651   0.203275
46  0.956140        9.0        26.0               33.0              27.0   0.537078          0.760153    126.0    7.922284  38.647277
47  0.938596       10.0        29.0              116.0              31.0   0.595065          0.529127    321.0    3.206805   9.529912
48  0.631579        8.0        25.0               21.0              42.0   0.733691          0.625671    479.0    9.999890   4.740624
49  0.964912        7.0        30.0               63.0              16.0   0.562973          0.832956    498.0    4.456060  25.287429
tunning time :  6.72
'''