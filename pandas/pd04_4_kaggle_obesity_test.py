# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
seed = 42

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
                
X, y = train_csv.drop(["NObeyesdad"], axis=1), train_csv.NObeyesdad
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

rf = RandomForestClassifier()
    
# Hyperparameter Optimization
gsc = GridSearchCV(rf, param_grid={
        'max_depth': [2,4,8,12,16,20,24],#필수
        'random_state' : [seed],
    } , cv=kf, verbose=100, refit=True)
gsc.fit(X_train, y_train)
x_predictsion = gsc.best_estimator_.predict(X_test)

best_acc_score = accuracy_score(y_test, x_predictsion) 
print(
f"""
============================================
[best_acc_score : {best_acc_score}]
[Best params : {gsc.best_params_}]
[Best value: {gsc.best_score_}]
============================================
"""
)
###########################################
# 저장
predictions = lbe.inverse_transform(gsc.best_estimator_.predict(test_csv)) 
submission_csv = pd.read_csv(path + "sample_Submission.csv")
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(rf).__name__}_loss{round(gsc.best_score_,4)}_" 
file_path = path + f"submission_{save_time}.csv"
submission_csv.to_csv(file_path, index=False)
print("저장완료")
# predict
