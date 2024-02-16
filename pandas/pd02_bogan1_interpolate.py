'''
결측치처리
1. 행 또는 열 삭제
2. 임의의 값
평균값: mean()
중위 : median
0 : fillna
앞의값 : ffill
뒤의값 : bfill
특정값 :  777 (조건을 붙어서 넣는게 좋음)
기타등등

3. 보간: interpolate
4. 모델 :predict
5. 부스팅 계열: 통상 결측치 이상치에 대해 자유롭다.


'''

import pandas as pd
from datetime import datetime
import numpy as np

dates = ['2/16/2024','2/17/2024','2/18/2024',
         '2/19/2024','2/20/2024','2/21/2024']
dates = pd.to_datetime(dates) #날짜 형식으로 변환
print(dates)

print("=========================================")
ts = pd.Series([2, np.nan, np.nan, 
                8, 10, np.nan], index = dates)
print(ts)
print("=========================================")
ts = ts.interpolate()
print(ts)
'''
2024-02-16     2.0
2024-02-17     4.0
2024-02-18     6.0
2024-02-19     8.0
2024-02-20    10.0
2024-02-21    10.0
'''