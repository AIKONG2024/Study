from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train, x_test, axis=0)
x = np.concatenate([x_train, x_test], axis=0) #(70000, 28, 28)
print(x.shape) #(70000, 28, 28)
#######[실습]############
#pca, 0.95이상인 n_components는 몇개?
#0.95 이상
#0.99 이상
#0.999 이상
#1.0 일때 몇개?
###########################
# model = RandomForestClassifier(random_state=42)
# x = x.reshape(-1, 28*28)
# for idx in range(1, 28*28):
#     scaler = StandardScaler()
#     x_2 = scaler.fit_transform(x)
#     pca = PCA(n_components=idx)
#     x_3 = pca.fit_transform(x)
x = x.reshape(-1, 28*28)
scaler = StandardScaler()
x = scaler.fit_transform(x)
pca = PCA(n_components=28*28)
x = pca.fit_transform(x)
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
# pd_evr = pd.Series(evr_cumsum)
# print("0.95이상 개수:", len(pd_evr[pd_evr >= 0.95])) #0.95이상 개수: 631
# print("0.99이상 개수:", len(pd_evr[pd_evr >= 0.99])) #0.99이상 개수: 454
# print("0.999이상 개수:", len(pd_evr[pd_evr >= 0.999])) #0.999이상 개수: 299
# print("1.0이상 개수:", len(pd_evr[pd_evr >= 1.0])) #1.0이상 개수: 72
print("0.95 이상 개수 : ",len(list(filter(lambda x: x >= 0.95, evr_cumsum))))
print("0.99 이상 개수 : ",len(list(filter(lambda x: x >= 0.99, evr_cumsum))))
print("0.999 이상 개수 : ",len(list(filter(lambda x: x >= 0.999, evr_cumsum))))
print("1.0 이상 개수 : ",len(list(filter(lambda x: x >= 1.0, evr_cumsum))))

print(np.argmax(evr_cumsum >= 0.95) + 1) #154
print(np.argmax(evr_cumsum >= 0.99) + 1) #331
print(np.argmax(evr_cumsum >= 0.999) + 1) #486
print(np.argmax(evr_cumsum >= 1.0) + 1) #713


import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

# print(evr_cumsum )
    # x_train, x_test, y_train , y_test = train_test_split(
    #     x, y, shuffle= True, random_state=123, train_size=0.8,
    #     stratify= y
    # )

    # start_time = time.time()
    # model.fit(x_train, y_train)
    # end_time = time.time()
    # predict = model.predict(x_test)

    # print(f'''
    # pca n_components : {idx} 
    # score : {accuracy_score(y_test, predict)}
    # 걸린 시간 : {round(end_time - start_time ,2 )} 초
    # ''')
# import matplotlib.pyplot as plt
# plt.plot(evr_cumsum)
# plt.grid()
# plt.show()