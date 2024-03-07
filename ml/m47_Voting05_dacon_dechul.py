import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

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

x = train_csv.drop(["대출등급"], axis=1)
y = train_csv["대출등급"]

lbe = LabelEncoder()
y = lbe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8,stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore') #워닝 무시
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
    'random_state' : 42,
}


votings = ['soft', 'hard']
for voting in votings:
    # 2. 모델 구성
    xgb = XGBClassifier()
    xgb.set_params(**parameters, eval_metric = 'logloss')
    rf = RandomForestClassifier()
    lr = LogisticRegression()

    model = VotingClassifier(
        estimators=[('LR',lr), ('RF',rf), ('XGB', xgb)],
        voting=voting,
    )

    print(f'==========={voting} voting============')
    # 3. 훈련
    model.fit(x_train, y_train)

    # 4. 평가, 예측
    result = model.score(x_test, y_test)
    print("최종점수 :", result)
    x_predict = model.predict(x_test)
    acc = accuracy_score(y_test, x_predict)
    print("acc_score :", acc)

print("=================기존==================")
model_class = [xgb, rf, lr]

for model in model_class:
    class_name = model.__class__.__name__
    # 3. 훈련
    model.fit(x_train, y_train)

    # 4. 평가, 예측
    result = model.score(x_test, y_test)
    x_predict = model.predict(x_test)
    score = accuracy_score(y_test, x_predict)
    print("{0} 정확도 : {1:.4f}".format(class_name, score))
'''
===========soft voting============
최종점수 : 0.7388234072381744
acc_score : 0.7388234072381744
===========hard voting============
최종점수 : 0.6679474531387922
acc_score : 0.6679474531387922
=================기존==================       
XGBClassifier 정확도 : 0.6567
RandomForestClassifier 정확도 : 0.8091
LogisticRegression 정확도 : 0.4246
'''