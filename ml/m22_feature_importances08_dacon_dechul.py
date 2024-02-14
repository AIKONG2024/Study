import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

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

x = train_csv.drop('대출등급', axis=1)
y = train_csv['대출등급']

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1234567, stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]
for model in models:
    
    #훈련
    model.fit(x_train, y_train)

    #평가, 예측
    acc = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    f1 = f1_score(y_test, y_predict, average='macro') 
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] mode f1 : ", f1)
    print(type(model).__name__ ,":", model.feature_importances_)
    
    submission = model.predict(test_csv)
    submission = train_le.inverse_transform(submission)

    submission_csv['대출등급'] = submission

    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)

'''
[DecisionTreeClassifier] model acc :  0.8326756663205261
[DecisionTreeClassifier] mode f1 :  0.7707555816577415
DecisionTreeClassifier : [6.10223195e-02 3.43251975e-02 1.47989049e-02 5.61849792e-03
 3.59966203e-02 2.97540137e-02 2.55534574e-02 8.00022798e-03
 5.37291937e-03 4.12673104e-01 3.66234495e-01 3.50758891e-04
 2.99483506e-04]
[RandomForestClassifier] model acc :  0.8071304949809622
[RandomForestClassifier] mode f1 :  0.6708862991204009
RandomForestClassifier : [0.09896486 0.02847401 0.04728213 0.01703776 0.08136222 0.08849017
 0.06939389 0.02464282 0.01653482 0.26283667 0.26349293 0.00053508
 0.00095266]
[GradientBoostingClassifier] model acc :  0.7493942540671512
[GradientBoostingClassifier] mode f1 :  0.6819904067219633
GradientBoostingClassifier : [2.30739048e-02 1.09903654e-01 5.33734599e-05 7.74418972e-04
 1.69608774e-02 6.24455643e-03 1.23945923e-03 6.19153846e-03
 9.92226236e-04 3.87284477e-01 4.47207819e-01 7.31821148e-05
 5.12868618e-07]
[XGBClassifier] model acc :  0.852128764278297
[XGBClassifier] mode f1 :  0.7935011541972988
XGBClassifier : [0.04566256 0.40178183 0.01210583 0.01572398 0.03459626 0.01720359
 0.0135195  0.02570579 0.01861944 0.19270255 0.20053996 0.01187714
 0.00996152]
'''

