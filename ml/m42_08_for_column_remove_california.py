from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

#1. 데이터
load = fetch_california_housing()
x = load.data
y = load.target

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
    'early_stopping_rounds' : 100,
}
#2. 모델 구성
model = XGBRegressor()
model.set_params(**parameters, eval_metric = 'rmse')

#3. 훈련
model.fit(x_train, y_train, eval_set = [(x_test, y_test)],  verbose = 0)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :" ,result)
x_predict = model.predict(x_test)
acc = r2_score(y_test, x_predict)
print("acc_score :", acc)


# 초기 특성 중요도
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)

# 제거된 피처의 개수를 저장하는 변수
num_removed_features = 0

# 각 반복에서 피처를 추가로 제거하면서 성능 평가
for i in range(len(model.feature_importances_) - 1):
    removed_feature_indices = sorted_indices[:i+1] 
    
    print(f"제거된 인덱스: {removed_feature_indices}")
    
    # 해당 특성 제거
    x_train_removed = np.delete(x_train, removed_feature_indices, axis=1)
    x_test_removed = np.delete(x_test, removed_feature_indices, axis=1)
    
    # 제거된 피처의 개수를 누적
    num_removed_features += 1
    print(f"누적 삭제된 features: {num_removed_features}\n")   

    # 모델 재구성 및 훈련
    model.fit(x_train_removed, y_train, eval_set=[(x_train_removed, y_train), (x_test_removed, y_test)],
              verbose=0)
    
    # 모델 평가
    result = model.score(x_test_removed, y_test)
    print("최종점수 :" ,result)
    
    x_predict = model.predict(x_test_removed)
    acc = r2_score(y_test, x_predict)
    print("acc_score :", acc)
    
'''
최종점수 : 0.4806468725827088
acc_score : 0.4806468725827088
제거된 인덱스: [4]
누적 삭제된 features: 1

최종점수 : 0.49644789018570734
acc_score : 0.49644789018570734
제거된 인덱스: [4 1]
누적 삭제된 features: 2

최종점수 : 0.47939426229547766
acc_score : 0.47939426229547766
제거된 인덱스: [4 1 0]
누적 삭제된 features: 3

최종점수 : 0.47790040158605873
acc_score : 0.47790040158605873
제거된 인덱스: [4 1 0 5]
누적 삭제된 features: 4

최종점수 : 0.47919342711727586
acc_score : 0.47919342711727586
제거된 인덱스: [4 1 0 5 6]
누적 삭제된 features: 5

최종점수 : 0.4715650837738319
acc_score : 0.4715650837738319
제거된 인덱스: [4 1 0 5 6 9]
누적 삭제된 features: 6

최종점수 : 0.4748008674169347
acc_score : 0.4748008674169347
제거된 인덱스: [4 1 0 5 6 9 3]
누적 삭제된 features: 7

최종점수 : 0.447306007805219
acc_score : 0.447306007805219
제거된 인덱스: [4 1 0 5 6 9 3 7]
누적 삭제된 features: 8

최종점수 : 0.47135071532750017
acc_score : 0.47135071532750017
제거된 인덱스: [4 1 0 5 6 9 3 7 2]
누적 삭제된 features: 9

최종점수 : 0.28952768360281333
acc_score : 0.28952768360281333
최종점수 : 0.7785315144768333
acc_score : 0.7785315144768333
제거된 인덱스: [4]
누적 삭제된 features: 1

최종점수 : 0.7753402596188486
acc_score : 0.7753402596188486
제거된 인덱스: [4 3]
누적 삭제된 features: 2

최종점수 : 0.7720558524034482
acc_score : 0.7720558524034482
제거된 인덱스: [4 3 1]
누적 삭제된 features: 3

최종점수 : 0.7668298866611045
acc_score : 0.7668298866611045
제거된 인덱스: [4 3 1 7]
누적 삭제된 features: 4

최종점수 : 0.6788281264879119
acc_score : 0.6788281264879119
제거된 인덱스: [4 3 1 7 6]
누적 삭제된 features: 5

최종점수 : 0.6139494102198447
acc_score : 0.6139494102198447
제거된 인덱스: [4 3 1 7 6 2]
누적 삭제된 features: 6

최종점수 : 0.5570052791184017
acc_score : 0.5570052791184017
제거된 인덱스: [4 3 1 7 6 2 5]
누적 삭제된 features: 7

최종점수 : 0.46548422200376216
acc_score : 0.46548422200376216
'''