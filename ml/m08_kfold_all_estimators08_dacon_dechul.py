import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

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
x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
allAlgorithms = all_estimators(type_filter='classifier') #41개
# allAlgorithms = all_estimators(type_filter='regressor') #55개
best_score = 0
best_model = ""

# 모델구성
from sklearn.metrics import accuracy_score
for name, algorithm in allAlgorithms :
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kf)
        y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
        acc_score = accuracy_score(y_test, y_predict)
        # 모델
        if best_score < acc_score:
            best_score = acc_score
            best_model = name
        # 3. 훈련
        # 평가, 예측
        print(f"""
=========================================================
[{name}]
acc : {scores} 
평균 acc : {round(np.mean(scores),4)}
eval acc : // {acc_score}
        """)
    except:
        continue
    
print("="*60)
print("[The Best score] : ", best_score )
print("[The Best model] : ", best_model )
print("="*60)
'''
[AdaBoostClassifier] score :  0.5003807545863621
[BaggingClassifier] score :  0.8641052267220491
[BernoulliNB] score :  0.37362409138110075
[CalibratedClassifierCV] score :  0.4190377293181031
[DecisionTreeClassifier] score :  0.8335064035998615
[DummyClassifier] score :  0.2992731048805815
[ExtraTreeClassifier] score :  0.4444444444444444
[ExtraTreesClassifier] score :  0.6715818622360679
[GaussianNB] score :  0.30889581169955
[GradientBoostingClassifier] score :  0.7494634821737626
[HistGradientBoostingClassifier] score :  0.8212530287296642
[KNeighborsClassifier] score :  0.4166839736933195
[LabelPropagation] score :  0.40865351332641053
[LabelSpreading] score :  0.4101073035652475
[LinearDiscriminantAnalysis] score :  0.4038075458636206
[LinearSVC] score :  0.41204569055036344
[LogisticRegression] score :  0.5264797507788161
[LogisticRegressionCV] score :  0.528141225337487
[MLPClassifier] score :  0.8219453097957771
[NearestCentroid] score :  0.3099342332987193
[PassiveAggressiveClassifier] score :  0.3058497750086535
[Perceptron] score :  0.30958809276566285
[QuadraticDiscriminantAnalysis] score :  0.36469366562824507
[RandomForestClassifier] score :  0.807061266874351
[RidgeClassifier] score :  0.37618553132571825
[RidgeClassifierCV] score :  0.37611630321910694
[SGDClassifier] score :  0.4137763932156456
[SVC] score :  0.7266874350986501
============================================================
[The Best score] :  0.8641052267220491
[The Best model] :  BaggingClassifier
'''