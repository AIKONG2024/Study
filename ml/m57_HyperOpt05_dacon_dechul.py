import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

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

x = train_csv.drop(["대출등급"], axis=1)
y = train_csv["대출등급"]

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias= False)
x = pf.fit_transform(x)

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
best {'colsample_bytree': 0.8865956791283274, 'learning_rate': 0.006548881828335525, 'max_depth': 3.0, 'min_bin': 492.0, 'min_child_samples': 74.0, 'min_child_weight': 42.0, 'num_leaves': 25.0, 'reg_alpha': 37.3016213743942, 'reg_lambda': 3.1836071961166383, 'subsample': 0.5023424019613534}
      target  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  min_bin  reg_lambda  reg_alpha
0   0.788878        8.0        29.0               90.0              32.0   0.929946          0.747073    152.0    3.802728  43.484203
1   0.834986        6.0        31.0              115.0              21.0   0.919577          0.987169    133.0    2.362119   7.226424
2   0.808505        8.0        29.0               26.0              48.0   0.675648          0.733944     42.0    8.400176  36.133737
3   0.806480        8.0        30.0              164.0              17.0   0.663992          0.742078     91.0    9.969161  42.956491
4   0.824861        5.0        36.0              164.0              12.0   0.640962          0.544161    276.0    9.970659   2.929567
5   0.796822        4.0        28.0               91.0              22.0   0.824233          0.546082    294.0    7.394361  28.361416
6   0.818371        5.0        37.0              159.0              29.0   0.609354          0.708533    399.0    8.638848   1.609043
7   0.818319       10.0        30.0              150.0              31.0   0.905925          0.727193    245.0    4.123947  40.680523
8   0.823407        4.0        29.0              119.0              14.0   0.796215          0.788735    197.0    1.545408   4.560204
9   0.811049        7.0        34.0               14.0              24.0   0.834956          0.502241    378.0    8.556279  47.048154
10  0.809492       10.0        27.0              170.0              29.0   0.649914          0.555593    284.0    6.926555  40.905418
11  0.804403        5.0        36.0               14.0              18.0   0.835366          0.967618    297.0    1.250390  49.850973
12  0.811932        8.0        34.0              111.0              25.0   0.667748          0.712776    337.0    1.106506  24.172731
13  0.807259        7.0        40.0              156.0              30.0   0.550623          0.554385    423.0    9.674065   2.557954
14  0.820500        8.0        33.0              171.0               2.0   0.752811          0.714087     58.0    1.613257  31.513153
15  0.821226       10.0        33.0              182.0              35.0   0.992096          0.851572    198.0    5.157374  31.766569
16  0.795057        3.0        33.0               68.0              29.0   0.610737          0.834196    221.0    4.750571  33.280609
17  0.828911        9.0        37.0              148.0               2.0   0.874666          0.563384     33.0    7.220697  17.592179
18  0.821694        7.0        35.0               51.0              44.0   0.767545          0.616295    254.0    6.379288  22.448113
19  0.813749        4.0        32.0              165.0              34.0   0.754167          0.642172    294.0    4.821782  14.112257
20  0.493587        3.0        25.0               74.0              42.0   0.502342          0.886596    492.0    3.183607  37.301621
21  0.786697        9.0        24.0               83.0              42.0   0.503217          0.905368    476.0    3.177848  46.342100
22  0.788826        9.0        25.0               49.0              39.0   0.523178          0.927537    499.0    0.097536  49.449364
23  0.551223        3.0        24.0               71.0              48.0   0.510057          0.905754    483.0    3.266060  36.516310
24  0.539176        3.0        26.0               41.0              50.0   0.558670          0.875279    448.0    2.535946  36.965176
25  0.740173        3.0        26.0               34.0              44.0   0.562715          0.853091    438.0    2.370177  37.633362
26  0.809180        4.0        26.0               53.0              49.0   0.707736          0.800635    358.0    0.262291  27.815087
27  0.551586        3.0        25.0               37.0              39.0   0.592980          0.938283    447.0    5.951101  21.103745
28  0.795368        6.0        27.0              137.0              50.0   0.554434          0.874513    499.0    3.758487  28.760275
29  0.550444        4.0        40.0               94.0              39.0   0.976422          0.791298    339.0    2.417319  44.926395
30  0.803728        5.0        25.0               67.0              44.0   0.704395          0.984682    405.0    0.646870  39.462833
31  0.719715        6.0        28.0               78.0              36.0   0.574304          0.956553    461.0    2.995921  11.651328
32  0.730048        3.0        27.0              123.0               7.0   0.529073          0.999511    131.0    4.108401  34.206916
33  0.810115        6.0        30.0               24.0              47.0   0.714930          0.886515    379.0    1.981222  42.969316
34  0.807570        5.0        31.0              102.0              46.0   0.617690          0.766432    421.0    5.383394  19.149857
35  0.797549        4.0        28.0               58.0              41.0   0.534966          0.829781    327.0    2.703901  26.272037
36  0.722000        4.0        26.0               41.0              37.0   0.630870          0.818042    464.0    3.644445  30.457825
37  0.775534        3.0        24.0               24.0              42.0   0.589134          0.690211    496.0    4.436375  35.266252
38  0.803624        5.0        29.0              101.0              33.0   0.687906          0.750842    157.0    5.642723  38.654823
39  0.760008        3.0        31.0              126.0              50.0   0.508584          0.869261    392.0    0.793248  41.705673
40  0.785918        4.0        25.0               84.0              21.0   0.947159          0.923872    368.0    1.787347  48.044343
41  0.796459        5.0        29.0               64.0              27.0   0.730502          0.963467    445.0    7.774822  44.644027
42  0.800301        6.0        28.0              195.0              18.0   0.500291          0.764235    327.0    2.218411  37.412996
43  0.797238        7.0        27.0               17.0              12.0   0.650931          0.677201     12.0    6.569740  25.965769
44  0.737370        3.0        39.0               43.0              45.0   0.795632          0.999374    408.0    1.205992  33.223729
45  0.814996        4.0        30.0              109.0              23.0   0.877432          0.899032     89.0    3.458362  29.976395
46  0.816449        5.0        26.0               76.0              27.0   0.588214          0.947217    156.0    9.225363   9.181702
47  0.789034        4.0        32.0               60.0              32.0   0.545345          0.851436    306.0    2.785174  41.036070
48  0.800924        3.0        38.0               93.0              41.0   0.688192          0.812666    222.0    4.257896  15.830764
49  0.753051        5.0        24.0               31.0              48.0   0.632168          0.774173    477.0    0.498237  49.979819
tunning time :  171.12
'''