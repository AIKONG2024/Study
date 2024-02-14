import numpy as np
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

datasets= load_boston()
x = datasets.data
y = datasets.target

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
'''
[DecisionTreeRegressor] model r2 :  0.5675551727949752
[DecisionTreeRegressor] mode eval_r2 :  0.5675551727949752
DecisionTreeRegressor : [6.59188486e-02 1.06551465e-06 6.12213371e-03 1.65752954e-04
 1.28554687e-02 6.22752675e-01 1.30310876e-02 5.76018317e-02
 1.04888514e-03 4.42790640e-03 7.52503427e-03 7.07126597e-03
 2.01478044e-01]
[RandomForestRegressor] model r2 :  0.7016178072704706
[RandomForestRegressor] mode eval_r2 :  0.7016178072704706
RandomForestRegressor : [0.03855863 0.00212279 0.0067511  0.0010815  0.01662486 0.60511956
 0.01659688 0.04847665 0.00464183 0.01362161 0.01020689 0.01323692
 0.22296076]
[GradientBoostingRegressor] model r2 :  0.8339005409867847
[GradientBoostingRegressor] mode eval_r2 :  0.8339005409867847
GradientBoostingRegressor : [0.02178298 0.00054221 0.00287798 0.         0.02776442 0.53379109
 0.01152535 0.0600612  0.00306402 0.0123679  0.02332203 0.00936654
 0.29353428]
[XGBRegressor] model r2 :  0.8468010535461267
[XGBRegressor] mode eval_r2 :  0.8468010535461267
XGBRegressor : [0.02745513 0.00094338 0.02482656 0.0016566  0.04589863 0.5345288
 0.01379093 0.06063454 0.01354152 0.03465008 0.01637223 0.00858056
 0.21712103]
'''
