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
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
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

# x = train_csv.drop('대출등급', axis=1)
# y = train_csv['대출등급']

# #데이터 분류

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#모델 생성
model = RandomForestClassifier()
unique = np.unique(train_csv['대출등급'])
for idx in range(1,min(len(test_csv.columns), len(unique))) :
    x = train_csv.drop(columns='대출등급', axis=1)
    y = train_csv['대출등급']
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    s_test_csv = scaler.transform(test_csv)
    
    lda = LinearDiscriminantAnalysis(n_components=idx)
    x = lda.fit_transform(x,y)
    a_test_csv = lda.transform(s_test_csv)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1234567, stratify=y)
    
    #훈련
    model.fit(x_train, y_train)

    # 평가 예측
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    predict = model.predict(x_test)
    print(f'''
    lda n_components : {idx} 
    score : {accuracy_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')
    
    submission = model.predict(a_test_csv)
    submission = train_le.inverse_transform(submission)

    submission_csv['대출등급'] = submission

    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)
evr = lda.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

'''
[RandomForestClassifier] model acc :  0.8050536517826238
[RandomForestClassifier] mode f1 :  0.6794309633545724
    lda n_components : 1
    score : 0.32973347178954654
    걸린 시간 : 10.9 초


    lda n_components : 2
    score : 0.400969193492558
    걸린 시간 : 8.83 초


    lda n_components : 3
    score : 0.4528902734510211
    걸린 시간 : 8.12 초


    lda n_components : 4
    score : 0.4692973347178955
    걸린 시간 : 12.78 초


    lda n_components : 5 
    score : 0.5062651436483212
    걸린 시간 : 12.44 초
    

    lda n_components : 6 
    score : 0.5268258913118726
    걸린 시간 : 12.61 초

[9.60325077e-01 3.18634367e-02 6.88333114e-03 6.32077634e-04
 2.44231999e-04 5.18454105e-05]
1.0000000000000002
[0.96032508 0.99218851 0.99907184 0.99970392 0.99994815 1.        ]
'''

