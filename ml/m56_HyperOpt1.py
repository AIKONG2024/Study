import numpy as np
import hyperopt

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

print(hyperopt.__version__)

search_space = {'x1' : hp.quniform('x1', -10, 10, 1),
                'x2' : hp.quniform('x2', -15, 15, 1)}

# hp.quniform(label, low, high, q) : label로 지정된 입력 밧ㅇ 변수 검색 공간을 최솟값 low에서 최대값 high까지 q의 간격을 가지고 설정
# hp.uniform(label, low, high) : 최소값 low에서  최대값 high 까지 정규분포 형태의 검색 공간 설정
# hp.randint(label, upper) : 0부터 최대값 upper 까지 random한 정수값으로 검색 공간 설정.
# hp.loguniform(label, low, high) : exp(uniform(low, high)) 값을 반환하며, 반환값의 log 변환된 값은 정규분포 형태를 가지는 검색 공간 설정.

def objectuve_func(search_space):
    x1 = search_space['x1']    
    x2 = search_space['x2']
    return_value = x1**2 -20*x2 #최소값이 나오려면 x1이 0, x2가 크면 클수록.

    return return_value

# def objective(space):
#     model = SVC(
#                C = space['C'],
#                degree= space['degree'],
#                max_iter= space['max_iter']
#                )
#     return {'loss': -1, 'status' : STATUS_OK }

trials = Trials()

best_param = fmin(
    fn=objectuve_func,
    space=search_space,
    algo=tpe.suggest,  # 알고리즘, 디폴트
    max_evals=20, #20번 실행
    trials=trials,
    rstate=np.random.default_rng(seed=10),
    #   rstate=333,
)
# 재훈련
# print(best_param)
# print(trials.results)
# print(trials.vals)
# #평가 예측

#[실습] 이쁘게 나오게 만들기
#판다스 데이터 프레임 사용
# [{'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}, 
# {'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0,
# 'x2': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8
import pandas as pd

# losses = []
# for dict in trials.results:
#     losses.append(dict.get('loss'))
losses = [dict.get('loss') for dict in trials.results]
df = pd.DataFrame( {
    'target' : losses,
    'x1' : trials.vals['x1'],
    'x2' : trials.vals['x2']})
print(df)
'''
    target    x1    x2
0   -216.0  -2.0  11.0
1   -175.0  -5.0  10.0
2    129.0   7.0  -4.0
3    200.0  10.0  -5.0
4    240.0  10.0  -7.0
5    -55.0   5.0   4.0
6    209.0   7.0  -8.0
7   -176.0  -2.0   9.0
8    -11.0  -7.0   3.0
9    -51.0   7.0   5.0
10   136.0   4.0  -6.0
11   -51.0  -7.0   5.0
12   164.0  -8.0  -5.0
13   321.0   9.0 -12.0
14    49.0  -7.0   0.0
15  -300.0   0.0  15.0
16   160.0  -0.0  -8.0
17  -124.0   4.0   7.0
18   -11.0   3.0   1.0
19     0.0  -0.0   0.0
'''