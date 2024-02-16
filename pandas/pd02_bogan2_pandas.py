import pandas as pd
import numpy as np

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

#결측치 확인
print(data.isna().sum())
print(data.info()) #describute, info는 꼭 할 것
print("===================================")

#결측치 삭제
print(data.dropna(axis=0)) #행 default: axis=0
# print(data.dropna(axis=1)) #열
# data = data.dropna(axis=0)
print("===================================")

#2-1. 특정값 - 평균
data2 = data.fillna(data.mean())
# data2 = data.fillna(np.mean(data))
print(data2)
print("===================================")

#2-2. 특정값 - 중위값
data3 = data.fillna(data.median())
# data3 = data.fillna(np.median(data))
print(data3)
print("===================================")

#2-3. 특정값 - 0 / 임의의값 채우기
data4 = data.fillna(0)
print(data4)
data4_2 = data.fillna(777)
print(data4_2)

print("===================================")

#2-4. 특정값 - ffill
# data5 = data.fillna(data.ffill())
data5 = data.ffill()
print(data5)
print("===================================")

#2-5. 특정값 - bfill
data6 = data.bfill()
print(data6)
print("===================================")

##################특정컬럼만
data7 = data
data7['x1'] =data['x1'].fillna(data['x1'].mean())
data7['x4'] = data['x4'].fillna(data['x4'].median())
data['x2'] = data['x2'].ffill()

print(data7)
