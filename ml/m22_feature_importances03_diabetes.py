from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.72, random_state=335688) 


random_state=42
#모델구성
models = [DecisionTreeRegressor(random_state=random_state), RandomForestRegressor(random_state=random_state),
          GradientBoostingRegressor(random_state=random_state), XGBRegressor(random_state=random_state)]
for model in models :
    # 컴파일 훈련
    model.fit(x_train, y_train)

    # 평가 예측
    r2 = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    r2_pred = r2_score(y_test, y_predict)
    print(f"[{type(model).__name__}] model acc : ", r2)
    print(f"[{type(model).__name__}] eval_acc : ", r2_pred)
    print(type(model).__name__ ,":", model.feature_importances_)

'''
[DecisionTreeRegressor] model acc :  -0.09453850950310771
[DecisionTreeRegressor] eval_acc :  -0.09453850950310771
DecisionTreeRegressor : [0.08395176 0.01111842 0.25722328 0.08843502 0.03596335 0.04225097
 0.03394282 0.04108224 0.3240624  0.08196974]
[RandomForestRegressor] model acc :  0.5417122641170801
[RandomForestRegressor] eval_acc :  0.5417122641170801
RandomForestRegressor : [0.07667082 0.00824693 0.30029139 0.10352945 0.04796762 0.05333925
 0.05308568 0.03014753 0.25103184 0.07568951]
[GradientBoostingRegressor] model acc :  0.48306219712320886
[GradientBoostingRegressor] eval_acc :  0.48306219712320886
GradientBoostingRegressor : [0.07775247 0.00526533 0.30963107 0.10216542 0.02945392 0.04636396
 0.04230194 0.05376712 0.27321068 0.0600881 ]
[XGBRegressor] model acc :  0.43926424393880215
[XGBRegressor] eval_acc :  0.43926424393880215
XGBRegressor : [0.04016969 0.05271643 0.17691037 0.08796495 0.04375722 0.05306406
 0.04940666 0.06157132 0.33492526 0.09951415]
'''
