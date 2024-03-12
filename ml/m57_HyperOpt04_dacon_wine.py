from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

#1. 데이터
path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']

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
best {'colsample_bytree': 0.9335981505512809, 'learning_rate': 0.007758441179431308, 'max_depth': 9.0, 'min_bin': 476.0, 'min_child_samples': 89.0, 'min_child_weight': 42.0, 'num_leaves': 24.0, 'reg_alpha': 49.50804088409414, 'reg_lambda': 3.177848215455366, 'subsample': 0.9739756391056589}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.565455        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.617273        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.574545        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.569091        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.623636        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.577273        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.613636        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.579091       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.626364        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.576364        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.575455       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.574545        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.589091        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.583636        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.587273        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.587273       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.571818        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.611818        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.590909        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.605455        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.567273        9.0        24.0               80.0              42.0   0.531082          0.920152    492.0    3.183607  42.566523
21  0.546364        9.0        24.0               89.0              42.0   0.973976          0.933598    476.0    3.177848  49.508041
22  0.566364        9.0        25.0               48.0              39.0   0.997987          0.917942    137.0    0.097536  49.779019
23  0.556364        9.0        26.0               89.0              49.0   0.958672          0.798473    464.0    3.208257  45.141505
24  0.580000        9.0        26.0              129.0              50.0   0.968898          0.885470    497.0    3.011246  36.965176
25  0.574545       10.0        24.0               61.0              46.0   0.956597          0.803050    459.0    5.482632  46.277164
26  0.572727        9.0        26.0               93.0              40.0   0.875393          0.949657    441.0    0.286941  48.600398
27  0.570000        6.0        25.0              136.0              50.0   0.713301          0.662548    367.0    2.322792  37.670036
28  0.551818        8.0        27.0               34.0              37.0   0.947328          0.797551    479.0    3.783137  45.433269
29  0.547273        7.0        28.0               32.0              38.0   0.891872          0.872160    338.0    4.039140  27.547059
30  0.594545        6.0        31.0              197.0              45.0   0.885010          0.990505    329.0    6.183751  11.049034
31  0.592727        7.0        28.0               32.0              42.0   0.921633          0.878612    413.0    2.370677  20.522190
32  0.579091        6.0        28.0               73.0              34.0   0.992863          0.934924    341.0    4.120252  28.689301
33  0.590909        8.0        25.0               54.0              38.0   0.805960          0.890677    384.0    5.707897  17.456233
34  0.613636        7.0        30.0               40.0               7.0   0.858111          0.997436     90.0    0.666424  11.580245
35  0.595455        5.0        27.0               20.0              46.0   0.707378          0.766176    167.0    2.481634  26.392452
36  0.608182        6.0        40.0              100.0              42.0   0.902399          0.836569    434.0    3.825218   7.661403
37  0.580909        8.0        31.0               78.0              27.0   0.931059          0.973529    317.0    2.099576  39.339841
38  0.567273       10.0        29.0              123.0              36.0   0.718149          0.767102    361.0    4.273643  35.612953
39  0.589091        5.0        24.0              102.0              21.0   0.789799          0.870846    398.0    2.964693  27.888791
40  0.608182        3.0        29.0               24.0              47.0   0.853832          0.909861    237.0    9.227684   7.048146
41  0.610000        7.0        27.0               60.0              32.0   0.901476          0.948844    268.0    7.774822   0.224430
42  0.572727        4.0        30.0              138.0              33.0   0.971413          0.825667    449.0    5.992575  34.389503
43  0.609091        8.0        28.0              109.0              22.0   0.830395          0.857589    318.0    6.577280  19.581220
44  0.566364       10.0        25.0               12.0              15.0   0.935749          0.679554    172.0    3.559489  14.434914
45  0.585455        7.0        32.0               39.0              26.0   0.812355          0.818566    410.0    1.826727  42.470307
46  0.598182        6.0        39.0               85.0              18.0   0.785730          0.903923    352.0    0.767116  24.680230
47  0.580909        8.0        26.0               67.0              41.0   0.854711          0.746201    383.0    4.507305  30.027058
48  0.578182        9.0        31.0              196.0              10.0   0.998548          0.962066    110.0    5.175470  38.568979
49  0.583636        5.0        28.0              117.0              44.0   0.729946          0.770674    214.0    2.793931  22.381732
tunning time :  16.6
'''