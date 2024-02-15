# xgboost 와  그리드서치, 랜덤서치, Halving등을 사용
# n_estimators , learning_rate, max_depth, colsample_bytree

# tree_method = 'gpu_hist',
# predictor = 'gpu_predictor',
# gpu_id = 0,

# m31_2 번보다 성능을 좋게 만든다.

from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
x = np.concatenate([x_train, x_test], axis=0)  # (70000, 28, 28)
# print(x.shape) #(70000, 28, 28)
x = x.reshape(-1, 28 * 28)

# from sklearn.model_selection import GridSearchCV

parameters = {
    # "n_estimators": 100,
    # "max_depth": 7,
    # # "colsample_bytree": 0.6,
    # "learning_rate": 0.0,
    "tree_method": "hist",
    # #   "predictor" : ["gpu_predictor"],
    # #   "gpu_id" : [0],
    # "grow_policy" : "lossguide",
    # # "booster": "gbtree",
    "device": "cuda",
    # "objective" : "multi:softmax",
    # "eval_metric" : "merror"
}

# n_estimators , learning_rate, max_depth, colsample_bytree

# tree_method = 'gpu_hist',
# predictor = 'gpu_predictor',
# gpu_id = 0,
# pca = PCA(n_components=154)
# pca_x = pca.fit_transform(x)

# x_train = pca_x[:60000]
# x_test = pca_x[60000:]
# model = GridSearchCV(xgb, param_grid=parameters, cv=5 , n_jobs=-1, refit=True, verbose=1)
# model.fit(x_train, y_train)

# from sklearn.metrics import accuracy_score
# best_predict = model.best_estimator_.predict(x_test)
# best_acc_score = accuracy_score(y_test, best_predict)

# print("best_model_acc_score : ", best_acc_score) #best_acc_score :  0.9333333333333333

# print(f'''
# 최적의 파라미터 :\t{model.best_estimator_}
# 최적의 매개변수 :\t{model.best_params_}
# best score :\t\t{model.best_score_}
# best_model_acc_score :\t{best_acc_score}
# ''')

# from sklearn.metrics import accuracy_score
# best_predict = model.best_estimator_.predict(x_test)
# best_acc_score = accuracy_score(y_test, best_predict)

model = XGBClassifier(**parameters)
components = [154, 331,486,713,784]
# components = [154]

for component in components:
    pca = PCA(n_components=component)
    pca_x = pca.fit_transform(x)

    x_train = pca_x[:60000]
    x_test = pca_x[60000:]

    # 컴파일 훈련
    start_time = time.time()
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    end_time = time.time()

    loss = model.score(x_test, y_test)
    predict = model.predict(x_test)
    print(f"PCA = {component}")
    print("acc score :", accuracy_score(y_test, predict))
    print("걸린시간 : ", round(end_time - start_time, 2), "초")
