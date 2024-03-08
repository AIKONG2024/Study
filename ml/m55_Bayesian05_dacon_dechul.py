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
from bayes_opt import BayesianOptimization
import time

#2. 모델
bayesian_params = {
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1,50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'min_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, min_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight': int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),
        'colsample_bytree' : colsample_bytree,
        'min_bin' : max(int(round(min_bin)), 10),
        'reg_lambda':max(reg_lambda, 0),
        'reg_alpha' : reg_alpha,
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

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds= bayesian_params,
    random_state=42,
)

n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print("bayesian tunning time : ", round(end_time - start_time, 2))
'''
{'target': 0.8327015940599201, 'params': {'colsample_bytree': 0.8431713813543159, 'learning_rate': 0.4217749296171438, 'max_depth': 5.728769751734323, 'min_bin': 181.4130784499443, 'min_child_samples': 145.82817876819948, 'min_child_weight': 42.51177615079155, 'num_leaves': 38.86982873460032, 'reg_alpha': 8.385637227215529, 'reg_lambda': 7.1262018360024335, 'subsample': 0.9840235407259854}}
bayesian tunning time :  360.6
'''