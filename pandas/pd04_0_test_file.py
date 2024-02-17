import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

seed = 42
def outliers(data_out):
    q1, q3 = np.percentile(data_out,[25, 75])
    iqr = q3 - q1
    
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    return (data_out > upper_bound) | (data_out < lower_bound)

##########################################
# 1. 데이터
path = "C:\_data\dacon\ddarung\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

print(train_csv.shape) #(1459, 11)
print(test_csv.shape) #(715, 10)

# 보간법 - 결측치 처리
## 1. interpolate
# train_csv.interpolate(inplace=True)
# test_csv.interpolate(inplace=True)

## 2. pandas
# train_csv = train_csv.dropna()
# train_csv = train_csv.fillna(0)
# train_csv = train_csv.fillna(train_csv.mean())
# train_csv = train_csv.fillna(train_csv.median())
# train_csv = train_csv.ffill()
train_csv = train_csv.bfill() #best

## 3. scikitlearn Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
#KNN
# imputer = KNNImputer(weights='distance')
# train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
# test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)
#round robin
# imputer = IterativeImputer(random_state=seed)
# train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
# test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)

# 이상치 확인
# print(train_csv.columns)
# 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object'
# import matplotlib.pyplot as plt
# plt.boxplot(train_csv)
# plt.show()

# 이상치 제거
hour_bef_precipitation_indexes = outliers(train_csv['hour_bef_precipitation'])
hour_bef_windspeed_indexes = outliers(train_csv['hour_bef_windspeed'])
hour_bef_ozone_indexes = outliers(train_csv['hour_bef_ozone'])
hour_bef_pm10_indexes = outliers(train_csv['hour_bef_pm10'])
hour_bef_pm2_5_indexes = outliers(train_csv['hour_bef_pm2.5'])
y_indexes = outliers(train_csv['count'])

# train_csv = train_csv.loc[~hour_bef_precipitation_indexes]
# train_csv = train_csv.loc[~hour_bef_windspeed_indexes]
# train_csv = train_csv.loc[~hour_bef_ozone_indexes]
# train_csv = train_csv.loc[~hour_bef_pm10_indexes]
# train_csv = train_csv.loc[~hour_bef_pm2_5_indexes]
# train_csv = train_csv.loc[~y_indexes]

# 이상치 처리
# median_value = train_csv['hour_bef_precipitation'].median()
# train_csv['hour_bef_precipitation'].loc[hour_bef_precipitation_indexes] = median_value
# median_value = train_csv['hour_bef_ozone'].median()
# train_csv['hour_bef_ozone'].loc[hour_bef_ozone_indexes] = median_value
# median_value = train_csv['hour_bef_pm10'].median()
# train_csv['hour_bef_pm10'].loc[hour_bef_pm10_indexes] = median_value
# median_value = train_csv['hour_bef_pm2.5'].median()
# train_csv['hour_bef_pm2.5'].loc[hour_bef_pm2_5_indexes] = median_value

# 평가 데이터 분할
X = train_csv.drop(["count"], axis=1)
y = train_csv["count"]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=seed)

##########################################
# 2. 모델 구현 , 훈련
#test model
def test_model():
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    return rf

#train model
def train_model():
    parameters = {}
    xgbr = XGBRegressor(**parameters)
    xgbr.fit(X_train, y_train,eval_set=[(X_test, y_test)], verbose=False)
    return xgbr

# model = test_model()
model = train_model()
###########################################
# 3. 평가 예측
print("score : ", model.score(X_test, y_test))
X_predict = model.predict(X_test)
mse_ = mean_squared_error(X_predict, y_test)
print("mse : ", mse_)
r2 = r2_score(X_predict, y_test)
print("r2_score : ", r2)

y_submit = model.predict(test_csv) # count 값이 예측됨.
submission_csv = pd.read_csv(path + "submission.csv")
submission_csv['count'] = y_submit

###########################################
# 저장
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}_loss{mse_}score{r2}_" 
file_path = path + f"submission_{save_time}.csv"
submission_csv.to_csv(file_path, index=False)
print("저장완료")