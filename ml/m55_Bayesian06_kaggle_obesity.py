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
{'target': 0.9034200385356455, 'params': {'colsample_bytree': 0.7515386889815849, 'learning_rate': 0.15947297649216263, 'max_depth': 9.73654382665555, 'min_bin': 478.5944423781708, 'min_child_samples': 192.78634987647808, 'min_child_weight': 44.84751970031764, 'num_leaves': 24.6697701297474, 'reg_alpha': 0.8597744078507609, 'reg_lambda': 5.355255628960389, 'subsample': 0.6755899284918989}}
bayesian tunning time :  156.51
'''