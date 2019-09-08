# coding: utf-8

import pandas as pd

gc = "C:\\Users\\shuangyang.luo\\Documents\\Airconditioner_MRO\\Online_analysis\\script\\data\\Latest\\空调数据采集0831-0901.csv"

test_data = pd.read_csv(gc, encoding="GB18030", engine='python')
test_data["index"] = 1
print(test_data)
print(12345678901234567890)
