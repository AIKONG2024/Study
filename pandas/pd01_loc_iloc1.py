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
print("=======================================")
# print(df[0]) #에러
# print(df["031"]) #에러
print(df["종목명"]) #판다스의 기준은 열

#아모레를 출력하고 싶음
# print(df[4,0]) #key 에러
# print(df[df["종목명", "045"]]) #key 에러
print(df["종목명"]["045"]) #아모레  ==> 판다스 열 행

# loc : 인덱스를 기준으로 행 데이터 추출
# iloc : 행번호를 기준으로 행 데이터 추출 ==> 인트loc로 외우기
print("=======================================")
print(df.loc['045'])
print("=======================================")
# print(df.loc[3]) #키에러
print(df.iloc[3])
print("=======================================")
print(df.loc["023"])
print(df.iloc[4])
print(df.iloc[-1])

print("아모레 시가: ")

print(df.loc["045"]["시가"]) #3500
print(df.loc["045"].loc["시가"]) #3500
print(df.시가.loc["045"])#3500
print(df["시가"]["045"]) #3500
print(df.시가["045"])#3500
print(df.iloc[3,1]) #3500
print(df.iloc[3].iloc[1])#3500
print(df.iloc[3].loc["시가"])#3500
print(df.iloc[3]["시가"]) #3500
print(df.loc["045","시가"])#3500

#====================deprecated
print("[deprecated]")
print(df.loc["045"][1]) #3500 -deprecated
print(df.iloc[3][1]) #3500 -deprecated

print("======================")
print(df.시가[-2:])
print(df.iloc[[3,4], 1]) #리스트형태  첫번째 : 1
print(df.loc[["045","023"], "시가"]) #리스트 사용
print(df.loc["045":"023", "시가"])  #슬라이싱 사용
# print(df.iloc[3:5, "시가"])#error
# print(df.iloc[[3,4], "시가"]) #error
# print(df.loc[3:5, "시가"]) #error
# print(df.loc[["045","023"], 1])#error