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
path = "C:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + "test.csv")

print(train_csv.shape) #(1459, 11)
print(test_csv.shape) #(715, 10)

print(train_csv.head(30))

# correlation 히트맵
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns

# print(matplotlib.__version__)

# sns.set(font_scale = 0.8)
# sns.heatmap(data=test_csv.corr(), square=True, annot=True, cbar=True)
# plt.show()
train_csv.drop(['holiday'], axis=1)

# 보간법 - 결측치 처리
print(test_csv.isna().sum())

# 컬럼 생성
def make_date(data) : 
    dt = pd.DatetimeIndex(data['datetime'])
    data['date'] = dt.date
    data['day'] = dt.day
    data['month'] = dt.month
    data['year'] = dt.year
    data['hour'] = dt.hour
    data['dow'] = dt.dayofweek
    return data

train_csv = make_date(train_csv)
test_csv = make_date(test_csv)
train_csv.set_index(keys='datetime', inplace=True)
test_csv.set_index(keys='datetime', inplace=True)

# 평가 데이터 분할
X = train_csv.drop(["count"], axis=1)
y = train_csv["count"]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=seed)


## 3. scikitlearn Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
#KNN
imputer = KNNImputer(weights='distance')
print( train_csv.index)
train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)
#round robin
# imputer = IterativeImputer(random_state=seed)
# train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
# test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)

# 이상치 확인
print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
# import matplotlib.pyplot as plt
# plt.boxplot(train_csv)
# plt.show()

# 이상치 제거

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