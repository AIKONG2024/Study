import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
data = pd.DataFrame(
    [
        [2, np.nan, 6, 8, 10],
        [2, 4, np.nan, 8, np.nan],
        [2, 4, 6, 8, 10],
        [np.nan, 4, np.nan, 8, np.nan],
    ]
)

data = data.transpose()
data.columns = ["x1", "x2", "x3", "x4"]
print(data)

'''
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''
imputer = SimpleImputer()

data1 = imputer.fit_transform(data) #평균
print(data1)
print("===================================")
imputer = SimpleImputer(strategy='mean')
data2 = imputer.fit_transform(data) #평균
print(data2)
print("===================================")
imputer = SimpleImputer(strategy='median')
data3 = imputer.fit_transform(data) #중위
print(data3)
print("===================================")
imputer = SimpleImputer(strategy='most_frequent')
data4 = imputer.fit_transform(data) #가장 자주나오는 수
print(data4)
print("===================================")
imputer = SimpleImputer(strategy='constant')
data5 = imputer.fit_transform(data) #상수 #0
print(data5)
print("===================================")
imputer = SimpleImputer(strategy='constant', fill_value=777)
data6 = imputer.fit_transform(data) #상수 #777
print(data6)
print("===================================")
imputer = KNNImputer() #KNN 알고리즘으로 결측치 처리
data7 = imputer.fit_transform(data)
print(data7)
print("===================================")
imputer = IterativeImputer()
data8 = imputer.fit_transform(data)
print(data8)
print("===================================")

print(np.__version__) #1.26.3

#pip install impyute
from impyute.imputation.cs import mice
aaa = mice(data.values, n=10, seed = 42)
print(aaa)