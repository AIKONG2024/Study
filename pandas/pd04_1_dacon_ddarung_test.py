import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

# correlation 히트맵
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns

# print(matplotlib.__version__)

# sns.set(font_scale = 0.8)
# sns.heatmap(data=test_csv.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 보간법 - 결측치 처리
print(test_csv.isna().sum())
'''
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
'''
## 1. interpolate
# train_csv['hour_bef_temperature'].interpolate(inplace=True)
# train_csv['hour_bef_precipitation'].interpolate(inplace=True)
# train_csv['hour_bef_windspeed'].interpolate(inplace=True)
# train_csv['hour_bef_humidity'].interpolate(inplace=True)
# train_csv['hour_bef_visibility'].interpolate(inplace=True)
# train_csv['hour_bef_ozone'].interpolate(inplace=True)
# train_csv['hour_bef_pm10'].interpolate(inplace=True)
# train_csv['hour_bef_pm2.5'].interpolate(inplace=True)
# test_csv['hour_bef_temperature'].interpolate(inplace=True)
# test_csv['hour_bef_precipitation'].interpolate(inplace=True)
# test_csv['hour_bef_windspeed'].interpolate(inplace=True)
# test_csv['hour_bef_humidity'].interpolate(inplace=True)
# test_csv['hour_bef_visibility'].interpolate(inplace=True)
# test_csv['hour_bef_ozone'].interpolate(inplace=True)
# test_csv['hour_bef_pm10'].interpolate(inplace=True)
# test_csv['hour_bef_pm2.5'].interpolate(inplace=True)

## 2. pandas
# train_csv['hour_bef_temperature'].dropna(inplace=True)
# train_csv['hour_bef_precipitation'].dropna(inplace=True)
# train_csv['hour_bef_windspeed'].dropna(inplace=True)
# train_csv['hour_bef_humidity'].dropna(inplace=True)
# train_csv['hour_bef_visibility'].dropna(inplace=True)
# train_csv['hour_bef_ozone'].dropna(inplace=True)
# train_csv['hour_bef_pm10'].dropna(inplace=True)
# train_csv['hour_bef_pm2.5'].dropna(inplace=True)

# train_csv['hour_bef_temperature'].fillna(0, inplace=True)
# train_csv['hour_bef_precipitation'].fillna(0, inplace=True)
# train_csv['hour_bef_windspeed'].fillna(0, inplace=True)
# train_csv['hour_bef_humidity'].fillna(0, inplace=True)
# train_csv['hour_bef_visibility'].fillna(0, inplace=True)
# train_csv['hour_bef_ozone'].fillna(0, inplace=True)
# train_csv['hour_bef_pm10'].fillna(0, inplace=True)
# train_csv['hour_bef_pm2.5'].fillna(0, inplace=True)
# test_csv['hour_bef_temperature'].fillna(0, inplace=True)
# test_csv['hour_bef_precipitation'].fillna(0, inplace=True)
# test_csv['hour_bef_windspeed'].fillna(0, inplace=True)
# test_csv['hour_bef_humidity'].fillna(0, inplace=True)
# test_csv['hour_bef_visibility'].fillna(0, inplace=True)
# test_csv['hour_bef_ozone'].fillna(0, inplace=True)
# test_csv['hour_bef_pm10'].fillna(0, inplace=True)
# test_csv['hour_bef_pm2.5'].fillna(0, inplace=True)

# train_csv['hour_bef_temperature'].fillna(train_csv.mean(), inplace=True)
# train_csv['hour_bef_precipitation'].fillna(train_csv.mean(), inplace=True)
# train_csv['hour_bef_windspeed'].fillna(train_csv.mean(), inplace=True)
# train_csv['hour_bef_humidity'].fillna(train_csv.mean(), inplace=True)
# train_csv['hour_bef_visibility'].fillna(train_csv.mean(), inplace=True)
# train_csv['hour_bef_ozone'].fillna(train_csv.mean(), inplace=True)
# train_csv['hour_bef_pm10'].fillna(train_csv.mean(), inplace=True)
# train_csv['hour_bef_pm2.5'].fillna(train_csv.mean(), inplace=True)
# test_csv['hour_bef_temperature'].fillna(test_csv.mean(), inplace=True)
# test_csv['hour_bef_precipitation'].fillna(test_csv.mean(), inplace=True)
# test_csv['hour_bef_windspeed'].fillna(test_csv.mean(), inplace=True)
# test_csv['hour_bef_humidity'].fillna(test_csv.mean(), inplace=True)
# test_csv['hour_bef_visibility'].fillna(test_csv.mean(), inplace=True)
# test_csv['hour_bef_ozone'].fillna(test_csv.mean(), inplace=True)
# test_csv['hour_bef_pm10'].fillna(test_csv.mean(), inplace=True)
# test_csv['hour_bef_pm2.5'].fillna(test_csv.mean(), inplace=True)

# train_csv['hour_bef_temperature'].fillna(train_csv.median(), inplace=True)
# train_csv['hour_bef_precipitation'].fillna(train_csv.median(), inplace=True)
# train_csv['hour_bef_windspeed'].fillna(train_csv.median(), inplace=True)
# train_csv['hour_bef_humidity'].fillna(train_csv.median(), inplace=True)
# train_csv['hour_bef_visibility'].fillna(train_csv.median(), inplace=True)
# train_csv['hour_bef_ozone'].fillna(train_csv.median(), inplace=True)
# train_csv['hour_bef_pm10'].fillna(train_csv.median(), inplace=True)
# train_csv['hour_bef_pm2.5'].fillna(train_csv.median(), inplace=True)
# test_csv['hour_bef_temperature'].fillna(test_csv.median(), inplace=True)
# test_csv['hour_bef_precipitation'].fillna(test_csv.median(), inplace=True)
# test_csv['hour_bef_windspeed'].fillna(test_csv.median(), inplace=True)
# test_csv['hour_bef_humidity'].fillna(test_csv.median(), inplace=True)
# test_csv['hour_bef_visibility'].fillna(test_csv.median(), inplace=True)
# test_csv['hour_bef_ozone'].fillna(test_csv.median(), inplace=True)
# test_csv['hour_bef_pm10'].fillna(test_csv.median(), inplace=True)
# test_csv['hour_bef_pm2.5'].fillna(test_csv.median(), inplace=True)

# train_csv['hour_bef_temperature'].ffill(inplace=True)
# train_csv['hour_bef_precipitation'].ffill(inplace=True)
# train_csv['hour_bef_windspeed'].ffill(inplace=True)
# train_csv['hour_bef_humidity'].ffill(inplace=True)
# train_csv['hour_bef_visibility'].ffill(inplace=True)
# train_csv['hour_bef_ozone'].ffill(inplace=True)
# train_csv['hour_bef_pm10'].ffill(inplace=True)
# train_csv['hour_bef_pm2.5'].ffill(inplace=True)
# test_csv['hour_bef_temperature'].ffill(inplace=True)
# test_csv['hour_bef_precipitation'].ffill(inplace=True)
# test_csv['hour_bef_windspeed'].ffill(inplace=True)
# test_csv['hour_bef_humidity'].ffill(inplace=True)
# test_csv['hour_bef_visibility'].ffill(inplace=True)
# test_csv['hour_bef_ozone'].ffill(inplace=True)
# test_csv['hour_bef_pm10'].ffill(inplace=True)
# test_csv['hour_bef_pm2.5'].ffill(inplace=True)

# train_csv['hour_bef_temperature'].bfill(inplace=True)
# train_csv['hour_bef_precipitation'].bfill(inplace=True)
# train_csv['hour_bef_windspeed'].bfill(inplace=True)
# train_csv['hour_bef_humidity'].bfill(inplace=True)
# train_csv['hour_bef_visibility'].bfill(inplace=True)
# train_csv['hour_bef_ozone'].bfill(inplace=True)
# train_csv['hour_bef_pm10'].bfill(inplace=True)
# train_csv['hour_bef_pm2.5'].bfill(inplace=True)
# test_csv['hour_bef_temperature'].bfill(inplace=True)
# test_csv['hour_bef_precipitation'].bfill(inplace=True)
# test_csv['hour_bef_windspeed'].bfill(inplace=True)
# test_csv['hour_bef_humidity'].bfill(inplace=True)
# test_csv['hour_bef_visibility'].bfill(inplace=True)
# test_csv['hour_bef_ozone'].bfill(inplace=True)
# test_csv['hour_bef_pm10'].bfill(inplace=True)
# test_csv['hour_bef_pm2.5'].bfill(inplace=True)


## 3. scikitlearn Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
#KNN
imputer = KNNImputer(weights='distance')
train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)
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

model = test_model()
###########################################
# 3. 평가 예측
print("score : ", model.score(X_test, y_test))
X_predict = model.predict(X_test)
mse_ = mean_squared_error(X_predict, y_test)
print("mse : ", mse_)
r2 = r2_score(X_predict, y_test)
print("r2_score : ", r2)
print("feature_importances : ", model.feature_importances_)