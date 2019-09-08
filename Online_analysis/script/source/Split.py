
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pandas as pd
from Online_analysis.script.source.Clean import data1

L_col = ["EnvTemp", "Left Temp Diff", "LEFT CONT CABIN DUCT", "LEFT L PACK", "LEFT SUPPLY DUCT"]
R_col = ["EnvTemp", "Right Temp Diff", "RIGHT FWD DUCT", "RIGHT AFT DUCT", "RIGHT L PACK", "RIGHT SUPPLY DUCT"]


# 切分数据集 # 暂时枚举 size = [0.1, 0.2, 0.3]
def split_(col, handle, size):
    X = data1[col]
    y = data1[handle]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    train = pd.concat([X_train, y_train], axis=1)
    train.dropna(inplace=True)
    test = pd.concat([X_test, y_test], axis=1)
    test.fillna(0, inplace=True)
    return X_train, X_test, y_train, y_test


X_train_left, X_test_left, y_train_left, y_test_left = split_(L_col, 'Left Handle', 0.1)
X_train_right, X_test_right, y_train_right, y_test_right = split_(R_col, 'Right Handle', 0.1)

# 重采样
def res(X_train, y_train, X_test, handle):
    X_t = pd.concat([X_train, y_train], axis=1)
    not_fraud = X_t[X_t[handle] == 0]
    fraud = X_t[X_t[handle] == 1]
    not_fraud_under_sampled = resample(not_fraud, replace=False, n_samples=len(fraud) * 10, random_state=27)
    under_sampled = pd.concat([not_fraud_under_sampled, fraud])
    y_train = under_sampled[handle]
    X_train = under_sampled.drop(handle, axis=1)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    return X_train, y_train, X_test


X_train_left, y_train_left, X_test_left = res(X_train_left, y_train_left, X_test_left, "Left Handle")
X_train_right, y_train_right, X_test_right = res(X_train_right, y_train_right, X_test_right, "Right Handle")

# 输出
# print(X_train_left)
# print(X_test_left)
# print(y_train_left)
# print(y_test_left)
# print(X_train_right)
# print(X_test_right)
# print(y_train_right)
# print(y_test_right)
# print(333333333333333333333333333333333)


