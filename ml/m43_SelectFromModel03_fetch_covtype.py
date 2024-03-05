from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

#1. 데이터
load = fetch_covtype()
x = load.data
y = load.target

lbe = LabelEncoder()
y = lbe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8,stratify=y)

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
    'early_stopping_rounds' : 100,
}
#2. 모델 구성
model = XGBClassifier()
model.set_params(**parameters, eval_metric = 'mlogloss')

#3. 훈련
model.fit(x_train, y_train, eval_set = [(x_test, y_test)],  verbose = 0)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :" ,result)
x_predict = model.predict(x_test)
acc = accuracy_score(y_test, x_predict)
print("acc_score :", acc)


# 초기 특성 중요도
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)

# 제거된 피처의 개수를 저장하는 변수
num_removed_features = 0

# 초기 특성 중요도

thresholds = np.sort(model.feature_importances_)

from sklearn.feature_selection import SelectFromModel

print("====================================================================")

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)# model 의 feature_importances_ 중 threshold 보다 같거나 높은 값만 살림
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(i, "\t변형된x_train : ", select_x_train.shape, "변형된 x_test :", select_x_test.shape)
    
    select_model =  XGBClassifier()
    select_model.set_params(
        **parameters,
    )

    select_model.fit(select_x_train, y_train, eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
              verbose=0)
    
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    
    print("Sel|| Select Threshold=%.3f, n=%d, ACC=%.2f" % (i, select_x_train.shape[1], score*100))