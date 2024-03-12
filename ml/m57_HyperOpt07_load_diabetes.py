from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

# 1. 데이터
load = load_diabetes()
x = load.data
y = load.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    BaggingRegressor,
    VotingClassifier,
    VotingRegressor,
    RandomForestRegressor,
    
)
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor
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

print("tunning time : ", round(end_time - start_time, 2))
'''
best {'colsample_bytree': 0.5270395579042814, 'learning_rate': 0.9217784590584764, 'max_depth': 5.0, 'min_bin': 80.0, 'min_child_samples': 32.0, 'min_child_weight': 11.0, 'num_leaves': 37.0, 'reg_alpha': 38.63955418933224, 'reg_lambda': 0.1405476710865269, 'subsample': 0.5603776725377438}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.468326        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.425865        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.475849        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.531074        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.368418        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.470487        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.493578        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.492659       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.448049        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.431114        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.464476       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.425885        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.513777        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.426732        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.342762        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.418838       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.379770        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.270389        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.434843        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.444878        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.307227        9.0        24.0              195.0               2.0   0.875409          0.650500    492.0    5.973511  17.374068
21  0.326647        9.0        24.0              198.0               3.0   0.877040          0.644998    476.0    6.009716  17.209311
22  0.317431        9.0        25.0              195.0               6.0   0.999546          0.607532     11.0    7.700980  10.698995
23  0.352105        9.0        40.0              138.0               9.0   0.965194          0.656347    475.0    2.939551  18.004318
24  0.287506        9.0        38.0              138.0               1.0   0.870332          0.598647    440.0    5.902892  20.364112
25  0.300418       10.0        38.0              134.0               8.0   0.956597          0.577090    347.0    5.588097  21.032299
26  0.421080        9.0        39.0              135.0              39.0   0.719569          0.508821    448.0    6.599356  10.794148
27  0.329589        6.0        38.0               96.0              12.0   0.872552          0.514139    118.0    7.736832  12.845895
28  0.207288        8.0        39.0               78.0               6.0   0.710606          0.597721    169.0    4.187837  27.295267
29  0.440491        7.0        39.0               72.0               7.0   0.700380          0.777709    160.0    3.343852  27.926656
30  0.193326        6.0        36.0               45.0               5.0   0.785701          0.585456     19.0    0.194694  37.531567
31  0.451464        6.0        36.0               34.0              18.0   0.518616          0.678343     82.0    2.298692  35.968723
32  0.209542        6.0        35.0               71.0               5.0   0.713818          0.920009    137.0    0.191942  36.732478
33  0.315774        6.0        39.0               49.0              14.0   0.784549          0.680873     10.0    0.641606  46.081914
34  0.024139        8.0        37.0               36.0              10.0   0.575272          0.526618    100.0    2.339811  27.841409
35  0.018426        5.0        37.0               32.0              11.0   0.560378          0.527040     80.0    0.140548  38.639554
36  0.311930        3.0        31.0               25.0              20.0   0.501598          0.529207    104.0    2.409968  44.032848
37  0.413207        5.0        37.0               57.0              11.0   0.562139          0.894269     70.0    2.146422  39.339111
38  0.451527        4.0        35.0               36.0              15.0   0.587230          0.755924     53.0    3.408505  33.452159
39  0.381328        5.0        32.0               25.0              21.0   0.540982          0.812158    190.0    0.544581  49.228486
40  0.295059        7.0        34.0               83.0              11.0   0.623486          0.529824    132.0    2.811805  30.387612
41  0.446470        5.0        30.0               13.0              27.0   0.510257          0.501235     97.0    1.676657  24.569031
42  0.451892        4.0        27.0               95.0              16.0   0.533764          0.618428    226.0    0.766044  42.655781
43  0.425979        3.0        37.0               60.0              22.0   0.569651          0.993461    178.0    4.038114  34.818701
44  0.374208        8.0        31.0              104.0              19.0   0.683106          0.680608    144.0    0.102903  26.746599
45  0.475626        7.0        40.0               18.0              23.0   0.588960          0.533067    113.0    1.193282  29.603130
46  0.568142        4.0        34.0               33.0              50.0   0.646249          0.742167     42.0    2.024374  39.497875
47  0.446943        8.0        28.0              120.0              27.0   0.590045          0.963090    320.0    2.837831  48.587302
48  0.147734        5.0        35.0               63.0               9.0   0.625011          0.563896    213.0    3.482606  45.621308
49  0.436162        8.0        38.0               39.0              14.0   0.681983          0.631992    273.0    1.009385   7.604492
tunning time :  2.63
'''