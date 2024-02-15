#스케일링 후 LCA후 분류 파일들 만들기
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
print(sk.__version__) #1.1.3

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) #(150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

#iris 는 ncomponent는 2 초과 해서 사용할 수 없음.
#n_component는 min(n_features, n_classes(라벨) - 1)로 결정한다.
lda =LinearDiscriminantAnalysis(n_components=1)
x = lda.fit_transform(x,y)
print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=42, shuffle=True, stratify=y
)

#2.모델
model = RandomForestClassifier(random_state=42)

#3.훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("=========================")
print(x.shape)
print('model.score : ', results)

'''
=========================
(150, 4)
model.score :  0.9333333333333333
=========================
(150, 3)
model.score :  0.9333333333333333
=========================
(150, 2)
model.score :  0.8666666666666667
=========================
(150, 1)
model.score :  0.8666666666666667

'''