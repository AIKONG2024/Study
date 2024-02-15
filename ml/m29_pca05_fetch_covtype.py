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
[RandomForestClassifier] model acc :  0.9532081879933909
[RandomForestClassifier] eval_acc :  0.9532081879933909
        pca n_components : 1      
    score : 0.4156339898207568
    걸린 시간 : 115.67 초     
    

    pca n_components : 2      
    score : 0.4975658331489268
    걸린 시간 : 71.74 초      
    

    pca n_components : 3     
    score : 0.772983206707482
    걸린 시간 : 59.97 초     
    

    pca n_components : 4     
    score : 0.921578028570726
    걸린 시간 : 87.6 초      
    

    pca n_components : 5      
    score : 0.9483538639325317
    걸린 시간 : 90.06 초      
    

    pca n_components : 6      
    score : 0.9530316441690639
    걸린 시간 : 94.98 초      
    

    pca n_components : 7      
    score : 0.9504376582823142
    걸린 시간 : 96.7 초       
    

    pca n_components : 8      
    score : 0.9434609426864351
    걸린 시간 : 98.34 초      
    

    pca n_components : 9
    score : 0.9472904526566841    
    걸린 시간 : 142.6 초


    pca n_components : 10
    score : 0.9421700966290477    
    걸린 시간 : 144.76 초


    pca n_components : 11
    score : 0.952515305746109     
    걸린 시간 : 141.34 초


    pca n_components : 12
    score : 0.9542056993926876    
    걸린 시간 : 139.9 초


    pca n_components : 13
    score : 0.9559206805832166    
    걸린 시간 : 139.31 초


    pca n_components : 14
    score : 0.9559083868112415    
    걸린 시간 : 139.2 초


    pca n_components : 15
    score : 0.9543286371124389    
    걸린 시간 : 139.59 초


    pca n_components : 16
    score : 0.9570086794030145    
    걸린 시간 : 183.17 초


    pca n_components : 17
    score : 0.9557854490914902    
    걸린 시간 : 187.58 초


    pca n_components : 18
    score : 0.954758919131568     
    걸린 시간 : 187.26 초


    pca n_components : 19
    score : 0.9559637087851295    
    걸린 시간 : 187.5 초


    pca n_components : 20
    score : 0.9547527722455804
    걸린 시간 : 189.51 초


    pca n_components : 21
    score : 0.9553674608443362
    걸린 시간 : 189.57 초


    pca n_components : 22
    score : 0.9549064443952694
    걸린 시간 : 188.03 초


    pca n_components : 23
    score : 0.9545745125519411
    걸린 시간 : 195.72 초


    pca n_components : 24
    score : 0.9538860613213346
    걸린 시간 : 185.96 초


    pca n_components : 25
    score : 0.9562157311106193
    걸린 시간 : 235.5 초


    pca n_components : 26
    score : 0.9553613139583487
    걸린 시간 : 240.96 초


    pca n_components : 27
    score : 0.9567628039635121
    걸린 시간 : 245.4 초


    pca n_components : 28
    score : 0.9557793022055027
    걸린 시간 : 247.49 초


    pca n_components : 29
    score : 0.9564370190061715
    걸린 시간 : 251.98 초


    pca n_components : 30
    score : 0.9568795947972757
    걸린 시간 : 255.72 초
'''