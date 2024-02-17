import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

seed = 42
def outliers(data_out):
    q1, q3 = np.percentile(data_out,[25, 75])
    iqr = q3 - q1
    
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    return (data_out > upper_bound) | (data_out < lower_bound)

##########################################
# 1. 데이터
path = "C:\_data\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

print(train_csv.shape) #(1459, 11)
print(test_csv.shape) #(715, 10)

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

# 보간법 - 결측치 처리
from sklearn.impute import KNNImputer
#KNN
imputer = KNNImputer(weights='distance')
train_csv = pd.DataFrame(imputer.fit_transform(train_csv), columns = train_csv.columns)
test_csv = pd.DataFrame(imputer.fit_transform(test_csv), columns = test_csv.columns)

# 평가 데이터 분할
X = train_csv.drop(["대출등급"], axis=1)
y = train_csv["대출등급"]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=seed)

##########################################
# 2. 모델 구현
#train model
def train_model():
    parameters = {
        "tree_method": "hist",
        "device": "cuda",
        "seed" : seed
    }
    xgbr = XGBClassifier(**parameters)
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
submission_csv = pd.read_csv(path + "sample_submission.csv")
submission_csv['count'] = y_submit

###########################################
# 저장
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}_loss{mse_}score{r2}_" 
file_path = path + f"submission_{save_time}.csv"
submission_csv.to_csv(file_path, index=False)
print("저장완료")