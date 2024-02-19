import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

seed = 42
def outliers(data_out):
    q1, q3 = np.percentile(data_out,[25, 75])
    iqr = q3 - q1
    
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    return (data_out > upper_bound) | (data_out < lower_bound)

##########################################
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
X = train_csv.drop(["count"], axis=1).drop(["casual"], axis=1).drop("registered", axis=1)
y = train_csv["count"]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=seed)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

parameters = {
    'n_estimators': [100, 200, 300, 400],  # 디폴트 100
    'learning_rate': [0.5, 1, 0.01, 0.001],  # 디폴트 0.3 / 0~1 / eta *
    'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 디폴트 0 / 0~inf
    # 'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],  # 디폴트 1 / 0~inf
    # 'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
    # 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
    # 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
    # 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],  # 디폴트 1 / 0~1
    # 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],  # 디폴트 0 / 0~inf/ L1 절대값 가중치 규제 / alpha
    # 'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10],  # 디폴트 1/ 0~inf/ L2 제곱 가중치 규제 / lambda
}

# 2. 모델
xgb = XGBRegressor(random_state = seed)
# Hyperparameter Optimization
model = GridSearchCV(xgb, param_grid=parameters , cv=kf, n_jobs=22)
model.fit(X_train, y_train)
x_predictsion = model.best_estimator_.predict(X_test)

results = model.score(X_test, y_test)
best_acc_score = r2_score(y_test, x_predictsion) 
print(
f"""
============================================
[best_acc_score : {best_acc_score}]
[Best params : {model.best_params_}]
[Best accuracy_score: {model.best_score_}]
============================================
"""
)