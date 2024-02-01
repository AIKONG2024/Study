from keras.preprocessing.text import Tokenizer
import numpy as np

text = "나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."

token = Tokenizer()
token.fit_on_texts([text]) 
print(token.word_index)
#{'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8} #Dictionary로 수치화 됨.

print(token.word_counts)
#OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

x = np.array(token.texts_to_sequences([text]))
print(x)#[[4, 2, 2, 3, 3, 5, 6, 7, 1, 1, 1, 8]]
print(x.shape)#(1, 12)

#to_categorical
# from keras.utils import to_categorical
# x1 = to_categorical(x)
# x1 = x1[:,:,1:]
# print(x1)
'''
[[[0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0.]
  [1. 0. 0. 0. 0. 0. 0. 0.]
  [1. 0. 0. 0. 0. 0. 0. 0.]
  [1. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1.]]]
'''

# 2. scikit-learn의 onehot
from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(sparse=False)
x1 = ohe.fit_transform(x.reshape(-1,1))
print(x1)
'''
[[0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1.]]
'''

# 3. pandas get_dummies()
# import pandas as pd
# x = x.reshape(-1*1) 
# x1 = pd.get_dummies(x)
# print(x1)
'''
        1      2      3      4      5      6      7      8
0   False  False  False   True  False  False  False  False
1   False   True  False  False  False  False  False  False
2   False   True  False  False  False  False  False  False
3   False  False   True  False  False  False  False  False
4   False  False   True  False  False  False  False  False
5   False  False  False  False   True  False  False  False
6   False  False  False  False  False   True  False  False
7   False  False  False  False  False  False   True  False
8    True  False  False  False  False  False  False  False
9    True  False  False  False  False  False  False  False
10   True  False  False  False  False  False  False  False
11  False  False  False  False  False  False  False   True
'''