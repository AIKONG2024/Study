import numpy as np
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import pandas as pd

datasets= load_boston()
x = datasets.data
y = datasets.target

# 넘파이로 삭제
# x = np.delete(x, 0, axis=1)
# 판다스로 삭제
x = pd.DataFrame(data=x, columns=datasets.feature_names)
x = x.drop(
    [
        "ZN",
        "CHAS",
        "B"
    ],
    axis=1,
)

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=20)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


random_state=42
#모델구성
models = [DecisionTreeRegressor(random_state=random_state), RandomForestRegressor(random_state=random_state),
          GradientBoostingRegressor(random_state=random_state), XGBRegressor(random_state=random_state)]
for model in models:
    model.fit(x_train, y_train)

    #평가 예측
    from sklearn.metrics import r2_score
    r2 = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    r2_pred = r2_score(y_test, y_predict)
    print(f"[{type(model).__name__}] model r2 : ", r2)
    print(f"[{type(model).__name__}] mode eval_r2 : ", r2_pred)
    print(type(model).__name__ ,":", model.feature_importances_)
    
    # 25%이하 컬럼
    # feature_importance_set = pd.DataFrame({'feature': x.columns, 'importance':model.feature_importances_})
    # feature_importance_set.sort_values('importance', inplace=True)
    # delete_0_25_features = feature_importance_set['feature'][:int(len(feature_importance_set.values) * 0.25)]
    # delete_0_25_importance = feature_importance_set['importance'][:int(len(feature_importance_set.values) * 0.25)]
    # print(f'''
    # 제거 할 컬럼명 : 
    # {delete_0_25_features}  
    # 제거 할 feature_importances_ : 
    # {delete_0_25_importance}    
    # ''')
'''
[XGBRegressor] model r2 :  0.8468010535461267
[XGBRegressor] mode eval_r2 :  0.8468010535461267
===================
제거후
[XGBRegressor] model r2 :  0.8403262657155619
[XGBRegressor] mode eval_r2 :  0.8403262657155619
'''
