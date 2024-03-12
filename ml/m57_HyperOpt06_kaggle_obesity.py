import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

# get data
path = "C:/_data/kaggle/obesity/"
train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + "test.csv")

# train_csv = colume_preprocessing(train_csv)

train_csv['BMI'] =  train_csv['Weight'] / (train_csv['Height'] ** 2)
test_csv['BMI'] =  test_csv['Weight'] / (test_csv['Height'] ** 2)

lbe = LabelEncoder()
cat_features = train_csv.select_dtypes(include='object').columns.values
for feature in cat_features :
    train_csv[feature] = lbe.fit_transform(train_csv[feature])
    if feature == "CALC" and "Always" not in lbe.classes_ :
        lbe.classes_ = np.append(lbe.classes_, "Always")
    if feature == "NObeyesdad":
        continue
    test_csv[feature] = lbe.transform(test_csv[feature]) 
                
x, y = train_csv.drop(["NObeyesdad"], axis=1), train_csv.NObeyesdad

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias= False)
x = pf.fit_transform(x)

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
best {'colsample_bytree': 0.9263812298699687, 'learning_rate': 0.001212195102726965, 'max_depth': 7.0, 'min_bin': 78.0, 'min_child_samples': 196.0, 'min_child_weight': 46.0, 'num_leaves': 27.0, 'reg_alpha': 17.92748846664006, 'reg_lambda': 0.7141962782760611, 'subsample': 0.5031764698178389}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.873314        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.902216        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.890414        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.883430        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.900048        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.892582        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.901975        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.887765       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.902216        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.884152        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.885597       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.885597        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.886561        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.890173        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.891378        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.891378       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.892341        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.897640        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.896676        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.898121        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.884875        9.0        24.0               80.0              42.0   0.531082          0.920152    492.0    3.183607  42.566523
21  0.886079        9.0        26.0              197.0               6.0   0.973976          0.656471    112.0    0.155584  46.366260
22  0.855491        8.0        25.0              134.0              39.0   0.691470          0.793399    141.0    3.217312  37.734984
23  0.871146        6.0        25.0              130.0              39.0   0.702701          0.799605    149.0    3.208257  35.532615
24  0.885356        6.0        24.0              132.0              39.0   0.715022          0.881753    162.0    2.764244  38.035101
25  0.882466        6.0        25.0              135.0              50.0   0.712475          0.798333     10.0    5.491687  28.221348
26  0.859586        7.0        26.0              132.0              46.0   0.563104          0.922974     76.0    0.295996  36.827341
27  0.852601        7.0        27.0              196.0              46.0   0.503176          0.926381     78.0    0.714196  17.927488
28  0.897399        9.0        27.0              191.0              37.0   0.516707          0.932935    173.0    3.792851  12.138108
29  0.853324        7.0        28.0               96.0              50.0   0.586014          0.892534     12.0    2.217307  18.563003
30  0.892100        7.0        31.0               96.0              49.0   0.502881          0.995158      9.0    2.212739  19.734879
31  0.895472        6.0        28.0               46.0              43.0   0.599413          0.960441    103.0    2.161472  10.306511
32  0.897399        8.0        30.0               68.0              46.0   0.575242          0.887961     32.0    0.358475  15.318291
33  0.895231        5.0        27.0               38.0              47.0   0.505024          0.883460     70.0    0.779493   8.564149
34  0.892341        7.0        31.0              103.0              50.0   0.546990          0.836015    119.0    1.852559  20.529616
35  0.890414        6.0        28.0               77.0               9.0   0.618109          0.765974     11.0    4.003598  27.831955
36  0.898121        5.0        30.0               84.0              41.0   0.588582          0.996425    476.0    0.624537   5.928567
37  0.887524        8.0        29.0              121.0              35.0   0.631309          0.942963     49.0    2.614626  16.863524
38  0.894750        7.0        26.0               63.0              19.0   0.522460          0.901330    186.0    6.082386  25.944934
39  0.899326        3.0        29.0              100.0              44.0   0.575353          0.858489     86.0    0.011030  12.308777
40  0.895954        4.0        28.0               24.0              27.0   0.648111          0.978346    220.0    1.086061  20.124260
41  0.901012        9.0        31.0              144.0              15.0   0.730925          0.821072    325.0    3.505542   0.273778
42  0.893064        8.0        27.0              111.0              22.0   0.500291          0.759106    120.0    4.391022  23.925085
43  0.898603        6.0        30.0              180.0              33.0   0.798366          0.691145     28.0    1.854553   7.949045
44  0.897881        5.0        39.0               91.0              47.0   0.678507          0.957098    433.0    2.568064  21.826974
45  0.866811        7.0        32.0               57.0              37.0   0.531758          0.863747    364.0    1.123143  17.906463
46  0.895713       10.0        28.0              122.0              27.0   0.871228          0.908720     59.0    9.225363  25.861458
47  0.896676        8.0        24.0              109.0              32.0   0.554270          0.822090     95.0    7.987955  10.793324
48  0.890655        5.0        25.0               24.0              45.0   0.641879          0.739255    219.0    1.477334  30.889600
49  0.898121        7.0        29.0               37.0              41.0   0.936557          0.772574    242.0    5.716860  14.712753
tunning time :  83.1
'''