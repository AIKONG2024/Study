import numpy as np
import pandas as pd
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T
df = pd.DataFrame(aaa, columns = ['x1','x2'])

def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    iqr = quartile_3 - quartile_1 #데이터 셋에서 정상적인 범위가 어디인가? 3분위 자리면 안전할 것 같다. 뺴기: 범위를 구하기 위해서 
    lower_bound = quartile_1 - (iqr * 1.5) #최소 백분위수의 1.5배수(범위를 조금 더 늘리기 위함) 범위 아래
    upper_bound = quartile_3 + (iqr * 1.5)  #최대 백분위수의 1.5배수 범위 위
    return (data_out < lower_bound) | (data_out > upper_bound)
    
outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.show()
################
#이상치 제거
x1_outlier_indexs = outliers(df['x1'])
x2_outlier_indexs = outliers(df['x2'])

#drop 방식
print(df.drop(df.index[x1_outlier_indexs]))
print(df.drop(df.index[x2_outlier_indexs]))

#drop 외 방식
print(df.loc[~x1_outlier_indexs])
print(df.loc[~x2_outlier_indexs])


####과제
# pd04_1_따릉이.py
# pa04_2_kaggle_bike.py
# pd04_3_대출.py
# pd04_4_kaggle_obsity.py