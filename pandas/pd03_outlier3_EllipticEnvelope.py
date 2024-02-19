import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,700,8,9,10,11,12,50]).reshape(-1,1)
print(aaa.shape)

from sklearn.covariance import EllipticEnvelope
#Quertile, iqr의 개념이 아니고 다른 알고리즘
outliers = EllipticEnvelope(contamination=.3) #이상치로 간주되는 데이터 포인트의 비율 0.1 이면 10프로, 0.3이면 30프로

outliers.fit(aaa)
results = outliers.predict(aaa) #찾는것이기 때문에 transform이 아닌 predict
print(results)

