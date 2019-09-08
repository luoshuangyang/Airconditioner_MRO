
import os
import pandas as pd
import numpy as np
import statistics as sta
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from Online_analysis.script.source.Test_import import test_data


test_data.drop(u'执行反吹左侧 (机器输出结果)', axis=1, inplace=True)
test_data.drop(u'执行反吹右侧 (机器输出结果)', axis=1, inplace=True)
test_data.drop(u'换件', axis=1, inplace=True)

test_data.loc[test_data[u"执行反吹左侧"] == test_data[u"执行反吹左侧"], u"执行反吹左侧"] = 1
test_data.loc[test_data[u"执行反吹右侧"] == test_data[u"执行反吹右侧"], u"执行反吹右侧"] = 1

test_data[u"执行反吹左侧"].fillna(0, inplace=True)
test_data[u"执行反吹右侧"].fillna(0, inplace=True)

test_data[u"左温差"] = test_data[u"左边L PACK"] - test_data[u"环境温度(℃)"]
test_data[u"右温差"] = test_data[u"右边L PACK"] - test_data[u"环境温度(℃)"]

test_data.drop_duplicates(subset=[u'日期', u'飞机号'], keep='first', inplace=True)
test_data.drop_duplicates(subset=['index', u'飞机号'], keep='last', inplace=True)

test_data.columns = ['Date', 'Location', 'PlaneNo', 'PlaneModel', 'EnvTemp', 'LEFT CONT CABIN DUCT', 'LEFT L PACK',
                     'LEFT SUPPLY DUCT', 'RIGHT FWD DUCT', 'RIGHT AFT DUCT', 'RIGHT L PACK', 'RIGHT SUPPLY DUCT',
                     'Left Handle', 'Right Handle', 'index', 'Left Temp Diff', 'Right Temp Diff']

test_data = pd.concat([test_data])

L_col = ["EnvTemp", "Left Temp Diff", "LEFT CONT CABIN DUCT", "LEFT L PACK", "LEFT SUPPLY DUCT"]
R_col = ["EnvTemp", "Right Temp Diff", "RIGHT FWD DUCT", "RIGHT AFT DUCT", "RIGHT L PACK", "RIGHT SUPPLY DUCT"]

X_test_left, y_test_left = test_data[L_col], test_data["Left Handle"]
X_test_right, y_test_right = test_data[R_col], test_data["Right Handle"]
X_test_left = StandardScaler().fit_transform(X_test_left)
X_test_right = StandardScaler().fit_transform(X_test_right)
grader_father = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+"..") \
                + '\\script\\source\\model'
L_model1 = joblib.load(os.path.join(grader_father, "L_model1.pkl"))
L_model2 = joblib.load(os.path.join(grader_father, "L_model2.pkl"))
L_model3 = joblib.load(os.path.join(grader_father, "L_model3.pkl"))
L_model4 = joblib.load(os.path.join(grader_father, "L_model4.pkl"))
L_model5 = joblib.load(os.path.join(grader_father, "L_model5.pkl"))
L_model6 = joblib.load(os.path.join(grader_father, "L_model6.pkl"))
L_model7 = joblib.load(os.path.join(grader_father, "L_model7.pkl"))
R_model1 = joblib.load(os.path.join(grader_father, "R_model1.pkl"))
R_model2 = joblib.load(os.path.join(grader_father, "R_model2.pkl"))
R_model3 = joblib.load(os.path.join(grader_father, "R_model3.pkl"))
R_model4 = joblib.load(os.path.join(grader_father, "R_model4.pkl"))
R_model5 = joblib.load(os.path.join(grader_father, "R_model5.pkl"))
R_model6 = joblib.load(os.path.join(grader_father, "R_model6.pkl"))
R_model7 = joblib.load(os.path.join(grader_father, "R_model7.pkl"))

print(L_model1)
print(L_model2)
print(L_model3)
print(L_model4)
print(L_model5)
print(L_model6)
print(L_model7)
print(R_model1)
print(R_model2)
print(R_model3)
print(R_model4)
print(R_model5)
print(R_model6)
print(R_model7)
L_pred1 = L_model1.predict(X_test_left)
L_pred2 = L_model2.predict(X_test_left)
L_pred3 = L_model3.predict(X_test_left)
L_pred4 = L_model4.predict(X_test_left)
L_pred5 = L_model5.predict(X_test_left)
L_pred6 = L_model6.predict(X_test_left)
L_pred7 = L_model7.predict(X_test_left)
#
R_pred1 = R_model1.predict(X_test_right)
R_pred2 = R_model2.predict(X_test_right)
R_pred3 = R_model3.predict(X_test_right)
R_pred4 = R_model4.predict(X_test_right)
R_pred5 = R_model5.predict(X_test_right)
R_pred6 = R_model6.predict(X_test_right)
R_pred7 = R_model7.predict(X_test_right)
# print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
y_pred_left = np.array([])
for i in range(0, len(X_test_left)):
    y_pred_left = np.append(y_pred_left, sta.mode([L_pred1[i], L_pred2[i], L_pred3[i], L_pred4[i], L_pred5[i]]))
#
y_pred_right = np.array([])
for i in range(0, len(X_test_right)):
    y_pred_right = np.append(y_pred_right, sta.mode([R_pred1[i], R_pred2[i], R_pred3[i], R_pred4[i], R_pred5[i]]))
#
test_data['Left Handle(machine output)'] = y_pred_left
test_data['Right Handle(machine output)'] = y_pred_right
print(y_pred_left)
print(y_pred_right)

