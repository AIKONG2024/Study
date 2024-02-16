import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

seed = 42

def outliers(data):
    q1, q3 = np.percentile(data,[25, 75])
    iqr = q3 - q1
    
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    return upper_bound, lower_bound

##########################################
# SCV
path = "C:\_data\dacon\ddarung\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

print(train_csv.shape) #(1459, 11)
print(test_csv.shape) #(715, 10)
print(train_csv.columns)

##########################################
# 보간 - 결측치 처리
train_csv.interpolate(inplace=True)
test_csv.interpolate(inplace=True)
##########################################
# 이상치 확인
lower_bound, upper_bound = outliers(train_csv)
##########################################
#이상치 제거

import matplotlib.pyplot as plt
plt.boxplot(train_csv)
plt.show()
##########################################
# 데이터 분할
X = train_csv.drop(["count"], axis=1)
y = train_csv["count"]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=seed)
##########################################
#test model
model = RandomForestRegressor()
model.fit(X_train, y_train)

##########################################
#xgboost
# 모델구성
# parameters = {
#     "learning_rate" : "0.0003"
# }
# xgbr = XGBRegressor(**parameters)
# xgbr.fit(X_train, y_train, eval = [X_train, y_train])

print("score : ", model.score(X_test, y_test))
X_predict = model.predict(X_test)
mse_ = mean_squared_error(X_predict, y_test)
print("mse : ", mse_)
print("r2_score : ", r2_score(X_predict, y_test))

y_submit = model.predict(test_csv) # count 값이 예측됨.
submission_csv = pd.read_csv(path + "submission.csv")
submission_csv['count'] = y_submit

######### submission.csv 만들기(count컬럼에 값만 넣어주면됨) ############
# import time as tm
# ltm = tm.localtime(tm.time())
# save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}_{mse_}_" 
# file_path = path + f"submission_{save_time}.csv"
# submission_csv.to_csv(file_path, index=False)
# print("저장완료")


# loss :  2632.7646484375
# r2 :  0.6287499373648455

# score :  0.7979421270465998
# mse :  1479.970470890411
# r2_score :  0.748340964562769
