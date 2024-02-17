import pandas as pd
import numpy as np
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
from sklearn.impute import KNNImputer
#KNN
imputer = KNNImputer(weights='distance')
train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)

# 이상치 처리
#이상치 처리에서의 개선점이 없어 사용하지 않음.

# 평가 데이터 분할
X = train_csv.drop(["count"], axis=1)
y = train_csv["count"]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=seed)

##########################################
# 2. 모델 구현
#train model
def train_model():
    parameters = {
        "learning_rate" : "0.090083",
        "max_depth" : 6,
        "gamma" : 0.03,
        "lambda" : 0.4,
        "alpha" : 0.5,
        "objective" : "reg:squarederror",
        "eval_metric" : "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "seed" : seed
    }
    xgbr = XGBRegressor(**parameters)
    xgbr.fit(X_train, y_train,eval_set=[(X_test, y_test)], verbose=False)
    return xgbr
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