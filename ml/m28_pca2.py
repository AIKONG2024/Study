#train_test_split후 스케일링 후 PCA 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
print(sk.__version__) #1.1.3

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) #(150, 4) (150,)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# pca = PCA(n_components=1) #n_components가 1 이면 선하나, 데이터손실이 많아질 수 있음. PCA 하기 전 Standart 스케일링 하는게 정설. 
# x = pca.fit_transform(x)
# print(x)
# print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=42, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

pca = PCA(n_components=1) 
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)


#2.모델
model = RandomForestClassifier(random_state=42)

#3.훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("=========================")
print(x_train.shape)
print('model.score : ', results)

'''
=========================
(120, 4)
model.score :  0.9666666666666667
=========================
(120, 3)
model.score :  0.3
=========================
(120, 2)
model.score :  0.3
=========================
(120, 1)
model.score :  0.3
'''