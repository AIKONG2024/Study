from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

datasets = fetch_covtype()

#모델 구현
model = RandomForestClassifier()

for idx in range(1,len(datasets.feature_names)) :
    x = datasets.data
    y = datasets.target
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)
    #컴파일, 훈련
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

'''
[LogisticRegression] model acc :  0.7229495593904902
[LogisticRegression] eval_acc :  0.7229495593904902
[KNeighborsClassifier] model acc :  0.9258249954103176
[KNeighborsClassifier] eval_acc :  0.9258249954103176
[DecisionTreeClassifier] model acc :  0.9352281072149807
[DecisionTreeClassifier] eval_acc :  0.9352281072149807
[RandomForestClassifier] model acc :  0.9532081879933909
[RandomForestClassifier] eval_acc :  0.9532081879933909
'''