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

x = x.drop(
    [
        "연체계좌수",
        "총연체금액",
        "근로기간"
    ],
    axis=1,
)
test_csv = test_csv.drop(
    [
        "연체계좌수",
        "총연체금액",
        "근로기간"
    ],
    axis=1,
)


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
    
    submission = model.predict(test_csv)
    submission = train_le.inverse_transform(submission)

    submission_csv['대출등급'] = submission

    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)
    

'''
[XGBClassifier] model acc :  0.852128764278297
[XGBClassifier] mode f1 :  0.7935011541972988
==================제거 후
[XGBClassifier] model acc :  0.8523364485981308
[XGBClassifier] mode f1 :  0.7873982311497212
'''

