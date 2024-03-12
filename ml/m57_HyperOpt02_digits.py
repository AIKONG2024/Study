from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

#1. 데이터
load_wine = load_digits()
x = load_wine.data
y = load_wine.target

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
best {'colsample_bytree': 0.7586727886962094, 'learning_rate': 0.38836435001362424, 'max_depth': 10.0, 'min_bin': 435.0, 'min_child_samples': 76.0, 'min_child_weight': 46.0, 'num_leaves': 25.0, 'reg_alpha': 47.12279423302162, 'reg_lambda': 3.135569593828902, 'subsample': 0.5316198805412711}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.866667        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.936111        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.836111        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.872222        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.941667        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.908333        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.930556        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.886111       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.944444        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.877778        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.858333       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.866667        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.913889        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.930556        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.925000        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.891667       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.855556        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.900000        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.902778        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.919444        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.766667        3.0        25.0               45.0              48.0   0.509515          0.875523     14.0    6.005167  35.216057
21  0.744444        9.0        24.0               35.0              49.0   0.510102          0.885442    481.0    5.856587  36.401605
22  0.772222        9.0        25.0               41.0              40.0   0.528146          0.913146    499.0    6.022670  36.672583
23  0.833333        3.0        24.0               65.0              49.0   0.500631          0.924586    465.0    3.190302  28.450016
24  0.519444        9.0        26.0               32.0              43.0   0.558706          0.873721    489.0    5.676399  49.645479
25  0.547222        9.0        26.0               29.0              40.0   0.568204          0.803050    499.0    0.308145  48.512336
26  0.566667        9.0        26.0               63.0              39.0   0.564559          0.800635    437.0    0.003846  48.737836
27  0.830556       10.0        27.0               11.0              41.0   0.709412          0.798202    359.0    3.640597  45.866426
28  0.700000        9.0        26.0               89.0              44.0   0.586683          0.941892    452.0    2.643539  44.193197
29  0.797222        6.0        28.0               28.0              37.0   0.705616          0.828329    493.0    0.531278  49.780117
30  0.883333        9.0        40.0              133.0              42.0   0.999043          0.771016    410.0    2.492050  40.137631
31  0.863889        7.0        31.0               80.0              45.0   0.703345          0.676722    326.0    7.760114   9.216631
32  0.913889        8.0        26.0               24.0               7.0   0.554762          0.978544    131.0    5.306327  17.996621
33  0.775000        6.0        28.0               54.0              36.0   0.616443          0.898571    386.0    3.911231  44.140762
34  0.816667        8.0        30.0               98.0              33.0   0.586652          0.997436    467.0    6.750557  38.708307
35  0.488889       10.0        25.0               76.0              46.0   0.531620          0.758673    435.0    3.135570  47.122794
36  0.844444       10.0        24.0               77.0              45.0   0.630870          0.759888    350.0    4.283473  28.484625
37  0.491667       10.0        25.0              119.0              46.0   0.533198          0.682060    435.0    8.065216  43.337732
38  0.872222       10.0        25.0              136.0              47.0   0.969035          0.674821    368.0    9.309546  42.231197
39  0.511111       10.0        31.0              195.0              38.0   0.524483          0.624771    152.0    7.800807  46.482180
40  0.886111        8.0        29.0              120.0              21.0   0.691508          0.585788    311.0    3.355719  39.018723
41  0.905556       10.0        27.0              105.0              27.0   0.727906          0.508451    432.0    8.915966  21.086882
42  0.613889        7.0        28.0              128.0              50.0   0.896599          0.684336    411.0    2.001569  42.258809
43  0.930556        5.0        30.0               90.0              11.0   0.647571          0.746169    388.0    2.827250   9.591954
44  0.919444        6.0        39.0              143.0              17.0   0.801708          0.654529    325.0    4.511346  26.715314
45  0.863889        8.0        29.0              104.0              27.0   0.586425          0.597606    269.0    7.920303  33.170766
46  0.869444       10.0        25.0               78.0              23.0   0.527022          0.520482    449.0    1.919366  30.151124
47  0.844444        4.0        27.0              123.0              32.0   0.679376          0.723514     91.0    8.314809  45.408002
48  0.702778        8.0        38.0              183.0              46.0   0.537098          0.700054    222.0    9.544817  38.174448
49  0.922222       10.0        32.0              113.0               6.0   0.936557          0.828070    169.0    0.741445  34.702892
tunning time :  14.34
'''