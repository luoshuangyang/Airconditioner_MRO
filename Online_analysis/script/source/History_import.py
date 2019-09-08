# coding: utf-8
import sys
import pandas as pd
from Online_analysis.script.conf.conf import ConfigFile

# 目前写死的 枚举
configfile = sys.path[0][:-6] + 'conf\\Conf.ini'
cf = ConfigFile(configfile)
gc1 = cf.GetPath('operation', 'Import1')
gc2 = cf.GetPath('operation', 'Import2')
gc3 = cf.GetPath('operation', 'Import3')
gc4 = cf.GetPath('operation', 'Import4')
gc5 = cf.GetPath('operation', 'Import5')
datalist = [gc1, gc2, gc3, gc4, gc5]

data = []
index = 1
for file_ in datalist:
    temp = pd.read_csv(file_, encoding="GB18030", engine='python')
    temp["index"] = index
    data.append(temp)
    index += 1
data_train = pd.concat(data)

# 输出
# print(data_train)
# print(1111111111111111111111111111111111111)

