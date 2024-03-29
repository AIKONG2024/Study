from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

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

parameters = {
    'n_estimators': 1000,  # 디폴트 100
    'learning_rate': 0.01,  # 디폴트 0.3 / 0~1 / eta *
    'max_depth': 3,  # 디폴트 0 / 0~inf
    'gamma': 0,
    'min_child_weight' : 0,
    'subsample' : 0.4,
    'colsample_bytree' :0.8,
    'colsample_bylevel' : 0.7,
    'colsample_bynode': 1,
    'reg_alpha': 0,
    'reg_lambda' : 1,
    'random_state' : 3377,
}
#2. 모델 구성
from sklearn.ensemble import BaggingRegressor
model = XGBRegressor()
model.set_params(**parameters, eval_metric = 'rmse')
model = BaggingRegressor(
    model,
    n_estimators=10,
    n_jobs=1,
    random_state=777,
    #   bootstrap=True, #디폴트, True : 중복을 허용
    bootstrap=False, 
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :" ,result)
x_predict = model.predict(x_test)
acc = r2_score(y_test, x_predict)
print("acc_score :", acc)

'''
기존
최종점수 : 0.7341108273476629
acc_score : 0.7341108273476629
bootstrap True
최종점수 : 0.722216257126677
acc_score : 0.722216257126677
bootstrap False
최종점수 : 0.7360390205961176
acc_score : 0.7360390205961176
'''