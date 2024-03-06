#그래프 그림
#1. value_counts x
#2. np.unique return_counts x
#3.groupby, count() 사용
# plt.bar 그림(quality 컬럼)
#힌트 : 데이터개수(y축) = 데이터 개수 ...

#[실습]
#1. 아웃라이어 확인, 처리
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

path = "C:/_data/dacon/wine/"

seed = 42
def outliers(data_out):
    q1, q3 = np.percentile(data_out,[25, 75])
    iqr = q3 - q1
    
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    return (data_out > upper_bound) | (data_out < lower_bound)

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})


print(train_csv.columns)

#이상치 제거 2, 5, 7, 8, 11
# for i,column in enumerate(train_csv.columns):
#     print("제거된 컬럼: ", column)
#     if i == 5 :
#         outlier_indexs = outliers(train_csv[column])
#         train_csv = train_csv.drop(train_csv.index[outlier_indexs])
    
x = train_csv.drop(columns='quality')
y = train_csv['quality']

#[실습] y의 클래스를 7개에서 5~3개로 줄여서 성능 비교
#총 9개
y = y-3
y_copy = y.copy()
print(np.unique(y))
for i, _ in enumerate(y_copy):
    if y_copy[i] == 0 or y_copy[i] == 1 or y_copy[i] == 2: 
        y[i] = 0
    elif y_copy[i] == 3 or y_copy[i] == 4 or y_copy[i] == 5: 
        y[i] = 3
    elif y_copy[i] == 6 or y_copy[i] == 7 or y_copy[i] == 8: 
        y[i] = 6
print(np.unique(y))
#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, random_state=seed, stratify=y)

scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)

#2. 모델 구성
model = RandomForestClassifier(random_state=seed)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :" ,result)
x_predict = model.predict(x_test)
acc = accuracy_score(y_test, x_predict)
print("acc_score :", acc)

f1 = f1_score(y_test, x_predict, average='macro')
print("f1_score : ", f1)

#라벨 변경전 원판 7개
# acc_score : 0.6627272727272727
# f1_score :  0.3857778567881957

#라벨 변경 후 원판 3개
# acc_score : 0.8254545454545454
# f1_score :  0.535974633183861
