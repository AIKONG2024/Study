from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

#1. 데이터
load = fetch_california_housing()
x = load.data
y = load.target

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias= False)
x = pf.fit_transform(x)

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
best {'colsample_bytree': 0.8919419226247984, 'learning_rate': 0.0022265979802540297, 'max_depth': 7.0, 'min_bin': 342.0, 'min_child_samples': 60.0, 'min_child_weight': 34.0, 'num_leaves': 24.0, 'reg_alpha': 8.45509711487804, 'reg_lambda': 3.6245294161047528, 'subsample': 0.9468962605167373}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.759068        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.832608        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.804714        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.804045        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.827923        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.821519        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.829539        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.822042       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.820496        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.811122        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.818194       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.817762        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.817435        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.760286        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.813120        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.802585       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.804370        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.807423        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.829072        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.825341        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.493620        9.0        40.0              199.0              41.0   0.500674          0.923757    483.0    3.183607  11.729206
21  0.826648        9.0        25.0              199.0              41.0   0.501555          0.905368    468.0    3.177848  10.681758
22  0.748625        9.0        40.0               89.0              39.0   0.972182          0.917942    137.0    0.103295  20.498764
23  0.731343        9.0        40.0              134.0              39.0   0.983397          0.932903    479.0    0.100472  18.998558
24  0.829816        9.0        39.0              133.0              49.0   0.525575          0.932269    500.0    2.497149  15.076594
25  0.814134       10.0        38.0              190.0              44.0   0.715448          0.867853    495.0    0.602131  18.468635
26  0.831483        9.0        39.0              139.0              38.0   0.582142          0.800635    444.0    0.031898  10.285757
27  0.826774        6.0        39.0              200.0              46.0   0.704072          0.955704    376.0    2.322792  27.889933
28  0.487801        8.0        24.0              129.0              36.0   0.949162          0.890709    469.0    3.455060  13.357094
29  0.241093        7.0        24.0               60.0              34.0   0.946896          0.891942    342.0    3.624529   8.455097
30  0.833101        6.0        24.0               49.0              35.0   0.937761          0.992020    334.0    5.769141   7.701249
31  0.829695        7.0        26.0               76.0              33.0   0.953592          0.783615    349.0    3.940738  14.990206
32  0.813182        8.0        24.0               32.0               7.0   0.902002          0.865163    433.0    3.099339   7.547904
33  0.249988        6.0        26.0               58.0              27.0   0.884495          0.827392    402.0    5.706205   0.457312
34  0.834820        6.0        26.0               52.0              19.0   0.866204          0.819556    406.0    5.951635   0.093490
35  0.810078        5.0        28.0               33.0              26.0   0.875597          0.760566    362.0    5.457230   0.116695
36  0.834539        6.0        25.0               90.0              14.0   0.807119          0.673666    318.0    7.814996   7.178255
37  0.818123        3.0        30.0               65.0              21.0   0.921645          0.825094    396.0    6.661418   5.257148
38  0.826851        4.0        27.0               97.0              27.0   0.893699          0.886778    181.0    9.420195   4.248741
39  0.829813        5.0        31.0               80.0              32.0   0.787729          0.759105    239.0    4.322256   0.172893
40  0.827008        7.0        28.0               26.0               6.0   0.848434          0.675238     99.0    2.446124  27.351812
41  0.822388        6.0        25.0               44.0              10.0   0.996702          0.975240    381.0    7.902973  37.132913
42  0.788565        5.0        26.0               66.0              15.0   0.966411          0.779654    269.0    4.533175   2.303892
43  0.837779        7.0        27.0               20.0              22.0   0.924303          0.951209    320.0    6.098765   9.721588
44  0.824055        4.0        29.0              103.0              24.0   0.816558          0.805535    442.0    5.474377   5.280144
45  0.802896        6.0        31.0               42.0              17.0   0.714563          0.842899    415.0    3.872201  16.266502
46  0.823890        7.0        29.0              120.0              28.0   0.851547          0.735590    218.0    5.243312  24.937609
47  0.834208        8.0        27.0               58.0              19.0   0.901982          0.994020    169.0    9.029999  12.599334
48  0.815092        5.0        25.0               77.0              32.0   0.780971          0.889770    308.0    7.759428  21.012367
49  0.839598        8.0        32.0               37.0              23.0   0.732510          0.591763    454.0    2.036096   3.356376
'''