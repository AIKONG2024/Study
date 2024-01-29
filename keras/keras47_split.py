import numpy as np

a = np.array(range(1, 11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):  # 5
        subset = dataset[i : (i + size)]  # subset 은 1~6/2~7/3~8/4~9/5~10 5개씩 나눠짐
        aaa.append(subset)  # 이어붙임
    return np.array(aaa)


bbb = split_x(a, size)
print(bbb)
print(bbb.shape)  # (6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape)  # (6, 4) (6,)
