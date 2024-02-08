import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold ,cross_val_predict
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

path = 'C:/_data/dacon/dechul/'
#데이터 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

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

x = train_csv.drop('대출등급', axis=1)
y = train_csv['대출등급']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234, stratify=y)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#데이터 분류
n_splits = 5 
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = BaggingClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[스케일링 + train_test_split]")
print("acc : ", scores, "\n평균 acc :", round(np.mean(scores),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
# print(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print("eval acc_ :", acc_score) #ecal acc_ : 0.9

'''
============================================================
[The Best score] :  0.8641052267220491
[The Best model] :  BaggingClassifier
============================================================
[kfold 적용 후]
acc :  [0.86312893 0.86660782 0.86577704 0.86250584 0.86031779]
평균 acc : 0.8637
============================================================
[StratifiedKFold 적용 후]
acc :  [0.86505011 0.86385586 0.86505011 0.86286931 0.86213522]
평균 acc : 0.8638
============================================================
[스케일링 + train_test_split]
acc :  [0.86551742 0.86691936 0.86665974 0.86203853 0.8652508 ]
평균 acc : 0.8653
eval acc_ : 0.8339506386513899
'''