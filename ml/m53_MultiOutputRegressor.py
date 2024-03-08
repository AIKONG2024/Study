import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

#1. 데이터
linnerud = load_linnerud()
x = linnerud.data
y = linnerud.target

print(x.shape, y.shape) #(20, 3) (20, 3)

# x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8)

models = [Ridge(), Lasso(), XGBRegressor(), LGBMRegressor(), CatBoostRegressor()]
for model in models:
    #2. 모델
    class_name = model.__class__.__name__ 
    if class_name == "LGBMRegressor" :
        model = MultiOutputRegressor(model)
    elif class_name == "CatBoostRegressor" :
        model = MultiOutputRegressor(model) #혹은 CatBoostRegressor 의 loss_function='MultiRMSE'
    model.fit(x,y)
    score = model.score(x, y)

    y_pred = model.predict(x)
    mae = mean_absolute_error(y, y_pred)
    print("{0} 스코어 : {1:.4f} mae : {2:.4f}".format(class_name, mae, score))

    print("=====================")
    print(model.predict([[2, 110, 43]])) #[[138.0005    33.002136  67.99897 ]]
    print("=====================")

#lightgbm은 y 컬럼이 여러개면 안먹힘.