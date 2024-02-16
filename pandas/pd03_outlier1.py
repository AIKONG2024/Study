import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])

def outliers(data_out):
    quartile_1, quartile_2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    
    print("1사분위 : ", quartile_1)
    print("q2 : ", quartile_2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1 #데이터 셋에서 정상적인 범위가 어디인가? 3분위 자리면 안전할 것 같다. 뺴기: 범위를 구하기 위해서 
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5) #최소 백분위수의 1.5배수(범위를 조금 더 늘리기 위함) 범위 아래
    upper_bound = quartile_3 + (iqr * 1.5)  #최대 백분위수의 1.5배수 범위 위
    print(lower_bound)
    print(upper_bound)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
    
outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

