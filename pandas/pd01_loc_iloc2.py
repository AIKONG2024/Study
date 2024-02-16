import pandas as pd

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
]

index = ["031", "032", "033", "045", "023"] #index는 문자열
columns = ["종목명", "시가", "종가"]
df = pd.DataFrame(data = data, index = index, columns = columns)

print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 032   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500
print("====================시가가 1100원 이상인 행을 모두 출력===================")
print(df[df.시가.astype(int) >= 1100])
print(df[df.시가 >= '1100'])
print(df.loc[df["시가"].astype(int) >= 1100])
print(df.loc[df["시가"] >= '1100'])
print(df.loc[df[df.시가.astype(int) >= 1100].index])
print(df.loc[df[df.시가 >= '1100'].index])


print("====================시가가 1100원 이상인 종가만 출력===================")
print(df.loc[df.시가 >= '1100']['종가'])
print(df.loc[df.시가 >= '1100','종가'])
print(df.loc[df[df.시가.astype(int) >= 1100].index, "종가"])
print(df.loc[df[df.시가 >= '1100'].index], "종가")
