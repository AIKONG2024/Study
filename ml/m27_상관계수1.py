from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return "XGBClassifier()"

# 1. 데이터
# x, y = datasets = load_iris(return_X_y=True)
datasets = load_iris()
x = datasets.data
y = datasets.target

df = pd.DataFrame(x, columns = datasets.feature_names)
print(df.head(20))
df['Target(Y)'] = y
print("====================상관계수 히트맵=====================")
print(df.corr())

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

print(matplotlib.__version__)

sns.set(font_scale = 1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()
#matplotlib 3.7.2에서 수치 잘 나옴.
#matplotlib 3.8.0에서는 수차 안나옴 . 버전 롤백.
