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
for idx in range(1,len(train_csv.columns)) :
    x = train_csv.drop(columns='대출등급', axis=1)
    y = train_csv['대출등급']
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    s_test_csv = scaler.transform(test_csv)
    
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    a_test_csv = pca.transform(s_test_csv)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1234567, stratify=y)
    
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
    
    submission = model.predict(a_test_csv)
    submission = train_le.inverse_transform(submission)

    submission_csv['대출등급'] = submission

    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)
evr = pca.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

'''
[RandomForestClassifier] model acc :  0.8050536517826238
[RandomForestClassifier] mode f1 :  0.6794309633545724

    pca n_components : 1
    score : 0.35943232952578746
    걸린 시간 : 10.08 초


    pca n_components : 2
    score : 0.41391484942886814
    걸린 시간 : 9.32 초


    pca n_components : 3
    score : 0.656282450674974
    걸린 시간 : 7.84 초


    pca n_components : 4 
    score : 0.8376600899965386
    걸린 시간 : 10.72 초
    

    pca n_components : 5
    score : 0.8319833852544133
    걸린 시간 : 10.95 초


    pca n_components : 6
    score : 0.8273451021114573
    걸린 시간 : 11.23 초


    pca n_components : 7
    score : 0.8161301488404292
    걸린 시간 : 11.47 초


    pca n_components : 8
    score : 0.8186915887850468
    걸린 시간 : 11.55 초


    pca n_components : 9
    score : 0.823399100034614
    걸린 시간 : 15.93 초


    pca n_components : 10
    score : 0.8177916233991
    걸린 시간 : 16.29 초


    pca n_components : 11
    score : 0.8109380408445829
    걸린 시간 : 16.64 초


    pca n_components : 12
    score : 0.8060228452751818
    걸린 시간 : 16.6 초


    pca n_components : 13
    score : 0.796123226029768
    걸린 시간 : 16.6 초

[9.91860985e-01 8.06466840e-03 6.53819950e-05 8.96447726e-06
 1.54188529e-10 1.29256024e-13 1.08948775e-14 6.72574164e-15
 1.65457269e-15 5.02910306e-16 6.57119730e-17 6.18162957e-17
 5.01763865e-19]
1.0
[0.99186098 0.99992565 0.99999104 1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.        ]
'''

