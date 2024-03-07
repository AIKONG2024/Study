import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

# get data
path = "C:/_data/kaggle/obesity/"
train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + "test.csv")

# train_csv = colume_preprocessing(train_csv)

train_csv['BMI'] =  train_csv['Weight'] / (train_csv['Height'] ** 2)
test_csv['BMI'] =  test_csv['Weight'] / (test_csv['Height'] ** 2)

lbe = LabelEncoder()
cat_features = train_csv.select_dtypes(include='object').columns.values
for feature in cat_features :
    train_csv[feature] = lbe.fit_transform(train_csv[feature])
    if feature == "CALC" and "Always" not in lbe.classes_ :
        lbe.classes_ = np.append(lbe.classes_, "Always")
    if feature == "NObeyesdad":
        continue
    test_csv[feature] = lbe.transform(test_csv[feature]) 
                
x, y = train_csv.drop(["NObeyesdad"], axis=1), train_csv.NObeyesdad

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
최종점수 : 0.8954720616570327
acc_score : 0.8954720616570327
===========hard voting============
최종점수 : 0.894990366088632
acc_score : 0.894990366088632
=================기존==================       
XGBClassifier 정확도 : 0.8916
RandomForestClassifier 정확도 : 0.8904
LogisticRegression 정확도 : 0.8324
'''