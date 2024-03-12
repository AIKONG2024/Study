from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

# 1. 데이터
path = "C:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

print(train_csv.shape) #(1459, 11)
print(test_csv.shape) #(715, 10)

# 보간법 - 결측치 처리
from sklearn.impute import KNNImputer
#KNN
imputer = KNNImputer(weights='distance')
train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)

# 평가 데이터 분할
x = train_csv.drop(["count"], axis=1).drop(["casual"], axis=1).drop("registered", axis=1)
y = train_csv["count"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

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
    
    model = XGBRegressor(**params, n_jobs = -1)
    
    model.fit(x_train, y_train,
              eval_set = [(x_train, y_train), (x_test,y_test)],
              eval_metric = 'mae',
              verbose=0,
              early_stopping_rounds = 50,
              )
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
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
'''
best {'colsample_bytree': 0.6192450411649614, 'learning_rate': 0.9434687890153647, 'max_depth': 8.0, 'min_bin': 318.0, 'min_child_samples': 172.0, 'min_child_weight': 36.0, 'num_leaves': 40.0, 'reg_alpha': 36.47179983608462, 'reg_lambda': 6.224282049493084, 'subsample': 0.533627972591073}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.326032        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.333401        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.300537        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.298585        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.325960        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.324231        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.325313        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.325216       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.312043        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.314541        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.317668       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.326699        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.331932        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.278736        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.317297        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.303220       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.294464        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.302820        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.320798        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.319351        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.291776        3.0        40.0              199.0              40.0   0.510558          0.858052    491.0    2.934197  15.113665
21  0.288293        6.0        40.0              196.0              40.0   0.511000          0.926690    483.0    2.928438  10.288361
22  0.293580        6.0        40.0              134.0              39.0   0.527284          0.917942    493.0    0.115026  10.054446
23  0.292419        6.0        24.0              200.0              49.0   0.548873          0.919566    448.0    5.945748   9.366747
24  0.310593        7.0        39.0              183.0              38.0   0.570275          0.665242    432.0    2.761836   0.039018
25  0.294146        9.0        38.0              135.0              43.0   0.507505          0.788355    459.0    0.092793  20.727867
26  0.297646        6.0        39.0              193.0              44.0   0.707736          0.899534    356.0    3.420459  12.712760
27  0.296824        7.0        24.0              184.0               7.0   0.558601          0.999751    420.0    5.536140   7.017732
28  0.286669        9.0        39.0              133.0              35.0   0.599042          0.790052    493.0    4.003712  18.450258
29  0.293204        9.0        38.0               96.0              34.0   0.592571          0.805855    328.0    4.128604  19.744746
30  0.301843        9.0        26.0               74.0              27.0   0.709472          0.592984    395.0    9.204498  26.420479
31  0.289022        9.0        39.0              127.0              47.0   0.703345          0.684851    462.0    7.852831  17.493161
32  0.294424        8.0        38.0              145.0              21.0   0.628881          0.772319    131.0    6.511146   5.483332
33  0.302074       10.0        36.0              107.0              32.0   0.990057          0.762286    426.0    2.084934  36.921656
34  0.298919        7.0        37.0               77.0              19.0   0.578511          0.819537    496.0    3.532655  27.352901
35  0.286674        8.0        35.0              124.0              37.0   0.539967          0.502866    373.0    9.315303   2.124812
36  0.342852        9.0        31.0              100.0              14.0   0.683465          0.884832    328.0    8.007086  12.587291
37  0.318578        5.0        39.0               50.0              27.0   0.623691          0.670955    402.0    0.685635  16.863524
38  0.299035        8.0        35.0              139.0              23.0   0.597205          0.966433    469.0    4.361343   4.330648
39  0.310321       10.0        38.0              158.0              46.0   0.787795          0.744743    159.0    5.556858  23.637993
40  0.316456        5.0        26.0               87.0              10.0   0.724773          0.585788    362.0    2.290494   0.723513
41  0.293422        7.0        30.0              117.0              42.0   0.647958          0.514663    438.0    3.857657  45.420978
42  0.319175        4.0        36.0              157.0              31.0   0.500291          0.645631    402.0    9.954347  29.466420
43  0.266409        8.0        40.0              172.0              36.0   0.533628          0.619245    318.0    6.224282  36.471800
44  0.302239        8.0        28.0              176.0              27.0   0.955819          0.533961    269.0    9.140369  37.342478
45  0.325137        7.0        37.0              191.0              16.0   0.882792          0.608137    310.0    7.652025  49.530502
46  0.291279        6.0        34.0              153.0              20.0   0.529031          0.570526    244.0    9.973437  35.071556
47  0.316016        8.0        40.0              143.0              29.0   0.813765          0.519023    169.0    8.388871  40.949964
48  0.312256        5.0        32.0              172.0              25.0   0.679389          0.624472    115.0    6.550515  43.453553
49  0.312858        7.0        36.0               28.0              37.0   0.574437          0.538621    225.0    7.000101  39.702407
'''