import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time


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

#모델 생성
model = RandomForestClassifier()
for idx in range(1,len(train_csv.columns)) :
    
    #훈련
    model.fit(x_train, y_train)

    # 평가 예측
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    predict = model.predict(x_test)
    print(f'''
    pca n_components : {idx} 
    score : {accuracy_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')
    
    submission = model.predict(test_csv)
    submission = train_le.inverse_transform(submission)

    submission_csv['대출등급'] = submission

    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)

'''
[RandomForestClassifier] model acc :  0.8050536517826238
[RandomForestClassifier] mode f1 :  0.6794309633545724
    pca n_components : 1
    score : 0.8114226375908619
    걸린 시간 : 8.06 초


    pca n_components : 2
    score : 0.8047767393561787
    걸린 시간 : 8.07 초


    pca n_components : 3
    score : 0.8077535479404638
    걸린 시간 : 8.07 초


    pca n_components : 4
    score : 0.8011768778123919
    걸린 시간 : 8.14 초


    pca n_components : 5
    score : 0.8067843544479059
    걸린 시간 : 8.03 초


    pca n_components : 6
    score : 0.7984077535479405
    걸린 시간 : 8.16 초


    pca n_components : 7
    score : 0.8127379716164763
    걸린 시간 : 7.92 초


    pca n_components : 8 
    score : 0.8044998269297334
    걸린 시간 : 7.96 초
    

    pca n_components : 9 
    score : 0.8071997230875736
    걸린 시간 : 8.0 초
    

    pca n_components : 10 
    score : 0.7949463482173763
    걸린 시간 : 7.79 초
    

    pca n_components : 11
    score : 0.804638283142956
    걸린 시간 : 7.97 초


    pca n_components : 12
    score : 0.8035306334371755
    걸린 시간 : 7.84 초


    pca n_components : 13
    score : 0.8037383177570093
    걸린 시간 : 7.95 초

'''

