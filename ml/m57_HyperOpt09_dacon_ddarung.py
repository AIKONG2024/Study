from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd
from sklearn.decomposition import PCA

# 1. 데이터
path = "C:\_data\dacon\ddarung\\"
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

# 이상치 처리
#이상치 처리에서의 개선점이 없어 사용하지 않음.

# 평가 데이터 분할
x = train_csv.drop(["count"], axis=1)
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
best {'colsample_bytree': 0.7883545238796842, 'learning_rate': 0.8072927613315389, 'max_depth': 9.0, 'min_bin': 459.0, 'min_child_samples': 135.0, 'min_child_weight': 43.0, 'num_leaves': 38.0, 'reg_alpha': 20.72786708664858, 'reg_lambda': 0.09279265626983513, 'subsample': 0.5075049877300963}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.733027        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.747971        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.686888        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.726545        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.741062        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.735941        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.737971        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.711159       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.703384        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.695106        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.739852       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.739761        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.747856        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.604530        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.732525        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.710266       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.684153        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.690446        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.714305        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.728107        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.660659        3.0        40.0              199.0              40.0   0.510558          0.858052    491.0    2.934197  15.113665
21  0.606718        6.0        40.0              196.0              40.0   0.511000          0.926690    483.0    2.928438  10.288361
22  0.609855        6.0        40.0              134.0              39.0   0.527284          0.917942    493.0    0.115026  10.054446
23  0.660322        6.0        24.0              200.0              49.0   0.548873          0.919566    448.0    5.945748   9.366747
24  0.705261        7.0        39.0              183.0              38.0   0.570275          0.665242    432.0    2.761836   0.039018
25  0.588524        9.0        38.0              135.0              43.0   0.507505          0.788355    459.0    0.092793  20.727867
26  0.641368        9.0        38.0              131.0              47.0   0.707736          0.781766    356.0    0.404915  27.143731
27  0.650867        9.0        38.0              138.0              44.0   0.598143          0.796986    435.0    9.204433  20.876062
28  0.648553        9.0        24.0               91.0              35.0   0.559596          0.671454    403.0    3.559485  17.019845
29  0.733520        7.0        39.0               66.0               7.0   0.976422          0.614357    464.0    5.677378  12.450569
30  0.649798        9.0        39.0              101.0              42.0   0.706294          0.506966    327.0    4.019877  19.338284
31  0.655797       10.0        26.0              127.0              46.0   0.703345          0.832013    414.0    7.886837   6.542949
32  0.666330        8.0        38.0              147.0              50.0   0.537526          0.879993    131.0    9.375174  36.871932
33  0.685153        7.0        36.0              184.0              19.0   0.503460          0.756318    499.0    2.161571   6.658249
34  0.702244        8.0        37.0               76.0              37.0   0.590344          0.674862    366.0    0.620432  27.305400
35  0.662821        5.0        31.0               99.0              32.0   0.636300          0.583823    324.0    9.315303   2.661082
36  0.713298        8.0        35.0              121.0              21.0   0.572287          0.751079    463.0    8.007086   0.255915
37  0.659331        9.0        39.0               34.0              27.0   0.619135          0.521459    398.0    3.494767  36.360059
38  0.673812        5.0        35.0              153.0              14.0   0.685294          0.964736    377.0    6.506115  24.224737
39  0.644407        7.0        36.0              137.0              23.0   0.500368          0.811309    159.0    2.061945  45.650824
40  0.749745        6.0        38.0              111.0              27.0   0.947159          0.885318    467.0    7.704972   5.067886
41  0.700000       10.0        37.0               82.0              10.0   0.729682          0.770108    420.0    9.842286  12.325599
42  0.711073        8.0        30.0              174.0              43.0   0.526197          0.695611    348.0    5.576126  38.661315
43  0.675485        4.0        26.0              156.0              31.0   0.798366          0.606735    384.0    6.764132  29.427375
44  0.677270        9.0        34.0              144.0              16.0   0.635178          0.646367    239.0    0.965705  49.940352
45  0.711193       10.0        32.0              191.0              36.0   0.882792          0.733492    306.0    4.453899  33.358345
46  0.730950        8.0        31.0              119.0              33.0   0.657141          0.988612     84.0    9.827666  21.387743
47  0.700402        5.0        40.0              108.0              30.0   0.580025          0.811851    172.0    8.345131  42.112008
48  0.625720        7.0        36.0              124.0              41.0   0.547837          0.536077    445.0    1.693876   3.828116
49  0.738334       10.0        28.0              159.0              26.0   0.776100          0.571133    272.0    0.018663  17.518740
'''