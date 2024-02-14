from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time

x,y = load_digits(return_X_y=True)

print(x.shape, y.shape) #(1797, 64) (1797,)
print(pd.value_counts(y, sort=False))

#Random으로 1번만 돌리고
#Grid Search, Randomized Search 로 돌려보기
#시간체크

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle= True, random_state=123, train_size=0.8,
    stratify= y
)
random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), XGBClassifier(random_state=random_state)]

for model in models :
    model.fit(x_train, y_train)

        #예측 평가
    acc = model.score(x_test, y_test)
    y_predict = model.predict(x_test)

    acc_pred = accuracy_score(y_test, y_predict)
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)
    print(type(model).__name__ ,":", model.feature_importances_)

'''
[DecisionTreeClassifier] model acc :  0.8833333333333333
[DecisionTreeClassifier] eval_acc :  0.8833333333333333
DecisionTreeClassifier : [0.         0.00077323 0.01302347 0.01394281 0.00230454 0.0632644
 0.00287201 0.         0.00153241 0.00511367 0.00736676 0.00154647
 0.01837601 0.00931932 0.         0.         0.         0.
 0.0135151  0.01902854 0.04319915 0.09551492 0.00154647 0.
 0.00151873 0.00403481 0.04981305 0.04895534 0.00624034 0.01686423
 0.01469382 0.         0.         0.05338646 0.024832   0.00077323
 0.07578661 0.02601307 0.01144927 0.         0.         0.00519172
 0.11979238 0.04876908 0.01609917 0.00780317 0.00891348 0.
 0.         0.00351637 0.00385552 0.0116559  0.00389458 0.01940003
 0.02539653 0.         0.         0.0023197  0.00303346 0.0044631
 0.05982064 0.00284925 0.00662571 0.        ]
[RandomForestClassifier] model acc :  0.9916666666666667
[RandomForestClassifier] eval_acc :  0.9916666666666667
RandomForestClassifier : [0.00000000e+00 2.40176326e-03 2.04293609e-02 9.91348309e-03
 8.74513706e-03 1.98022278e-02 9.93865962e-03 7.54194272e-04
 1.11284965e-04 8.18787396e-03 2.64103763e-02 7.45845819e-03
 1.69951735e-02 2.80396085e-02 5.35337856e-03 6.40284208e-04
 1.18543764e-04 6.96392418e-03 1.95603056e-02 2.76272461e-02
 2.81760012e-02 5.03685893e-02 8.66131560e-03 3.96936227e-04
 1.13496987e-04 1.37221199e-02 4.07788981e-02 2.43305802e-02
 2.94848520e-02 2.00302096e-02 3.26507642e-02 1.28944518e-05
 0.00000000e+00 2.88259273e-02 2.93590342e-02 1.84337871e-02
 4.38903595e-02 1.91578605e-02 2.51533730e-02 0.00000000e+00
 7.55475481e-05 1.10970596e-02 3.86298218e-02 4.22614609e-02
 2.46707759e-02 2.02052383e-02 1.82995303e-02 1.38312910e-04
 1.14841103e-04 2.38627763e-03 1.60405171e-02 2.26655472e-02
 1.27589411e-02 2.53046300e-02 2.75061492e-02 1.73339771e-03
 2.44200013e-05 2.01441883e-03 2.20440366e-02 1.05728989e-02
 2.17283962e-02 2.78084858e-02 1.56758159e-02 3.24519650e-03]
'''