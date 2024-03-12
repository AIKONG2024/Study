from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

#1. 데이터
load = fetch_covtype()
x = load.data
y = load.target

lbe = LabelEncoder()
y = lbe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8,stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore') #워닝 무시
import time
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

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
              eval_metric = 'mlogloss',
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
best {'colsample_bytree': 0.7957514587810519, 'learning_rate': 0.006084347137814083, 'max_depth': 4.0, 'min_bin': 466.0, 'min_child_samples': 136.0, 'min_child_weight': 10.0, 'num_leaves': 26.0, 'reg_alpha': 21.457442560088577, 'reg_lambda': 3.3338518894086735, 'subsample': 0.8914250194790228}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.792906        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.871957        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.912756        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.909279        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.837362        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.798361        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.810186        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.923806       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.829867        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.892077        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.882292       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.840546        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.828008        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.920166        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.919675        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.939451       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.823283        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.939201        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.881122        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.820478        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.832638        9.0        25.0               84.0              38.0   0.992496          0.920152    158.0    3.616822  27.278168
21  0.756461        3.0        25.0               87.0               6.0   0.940318          0.651939    497.0    0.155584  12.073893
22  0.763251        6.0        24.0               52.0              10.0   0.952797          0.653883    122.0    0.372233  10.464530
23  0.757115        3.0        24.0               43.0               7.0   0.959106          0.656373    493.0    0.524830  10.626806
24  0.753931        3.0        26.0               44.0               7.0   0.966445          0.667244    498.0    0.202349  20.025484
25  0.751538        3.0        26.0              128.0               6.0   0.991490          0.600703    491.0    2.414023  19.490884
26  0.765892        4.0        26.0              130.0               6.0   0.875703          0.600118    459.0    2.785603  19.072665
27  0.796201        3.0        27.0              198.0               1.0   0.989376          0.508285    453.0    2.552756  18.327364
28  0.719069        4.0        26.0              136.0              10.0   0.891425          0.795751    466.0    3.333852  21.457443
29  0.735076        5.0        28.0              134.0              15.0   0.892511          0.807196    347.0    3.503321  14.280090
30  0.849272        6.0        28.0              137.0              15.0   0.718832          0.790967    360.0    5.647933   7.701249
31  0.753053        5.0        31.0              199.0              20.0   0.893785          0.891849    333.0    3.453145  26.972245
32  0.754851        6.0        28.0               97.0              10.0   0.856922          0.773057    433.0    4.331601  15.201165
33  0.862370        4.0        30.0              185.0              17.0   0.800610          0.827157    396.0    6.143856   6.282305
34  0.874685        5.0        25.0              102.0              11.0   0.919795          0.892060    322.0    3.127913  22.472035
35  0.859677        6.0        27.0              136.0              21.0   0.707378          0.961115    456.0    1.945815  35.042624
36  0.858523        5.0        29.0               72.0              14.0   0.711822          0.874431    379.0    3.981222   0.406679
37  0.855838        4.0        31.0              118.0               4.0   0.510754          0.759262    424.0    5.560155  29.463565
38  0.731711        5.0        28.0              146.0              23.0   0.815037          0.808726    359.0    4.414190  38.478577
39  0.762562        4.0        29.0              145.0              39.0   0.782568          0.937833    391.0    7.965186  37.275661
40  0.865460        6.0        24.0              180.0              23.0   0.813022          0.688900    314.0    9.227684  46.799923
41  0.864332        5.0        39.0              110.0              26.0   0.850812          0.993931    356.0    4.583441  38.963444
42  0.867120        7.0        30.0              193.0              27.0   0.691877          0.749904    471.0    6.466104  42.701881
43  0.790487        4.0        27.0              122.0              19.0   0.733408          0.866416    417.0    3.025532  46.388399
44  0.832724        5.0        32.0              158.0              43.0   0.927228          0.813759    272.0    2.077869  49.713449
45  0.875347        6.0        25.0               76.0              33.0   0.611161          0.688748    230.0    5.539236  30.260009
46  0.802793        7.0        31.0              173.0              50.0   0.824773          0.732746    370.0    7.674822  26.076037
47  0.847749        4.0        29.0              103.0              37.0   0.782295          0.916272    439.0    0.679018  34.648169
48  0.817896        5.0        26.0              143.0              17.0   0.858790          0.777344    192.0    5.223663  44.979446
49  0.864436        9.0        27.0              153.0              13.0   0.903492          0.847604    410.0    1.009385  23.300240
tunning time :  478.27
'''