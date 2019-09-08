import pandas as pd
from Online_analysis.script.source.History_import import data_train


def clean(d):
    """数据预处理"""
    d.drop(u'执行反吹左侧 (机器输出结果)', axis=1, inplace=True)
    d.drop(u'执行反吹右侧 (机器输出结果)', axis=1, inplace=True)
    d.drop(u'换件', axis=1, inplace=True)
    d.loc[d[u"执行反吹左侧"] == d[u"执行反吹左侧"], u"执行反吹左侧"] = 1
    d.loc[d[u"执行反吹右侧"] == d[u"执行反吹右侧"], u"执行反吹右侧"] = 1
    d[u"执行反吹左侧"].fillna(0, inplace=True)
    d[u"执行反吹右侧"].fillna(0, inplace=True)
    d[u"左温差"] = d[u"左边L PACK"] - d[u"环境温度(℃)"]
    d[u"右温差"] = d[u"右边L PACK"] - d[u"环境温度(℃)"]
    d.drop_duplicates(subset=[u'日期', u'飞机号'], keep='first', inplace=True)
    d.drop_duplicates(subset=['index', u'飞机号'], keep='last', inplace=True)
    d.columns = ['Date', 'Location', 'PlaneNo', 'PlaneModel', 'EnvTemp', 'LEFT CONT CABIN DUCT', 'LEFT L PACK',
                 'LEFT SUPPLY DUCT', 'RIGHT FWD DUCT', 'RIGHT AFT DUCT', 'RIGHT L PACK', 'RIGHT SUPPLY DUCT',
                 'Left Handle', 'Right Handle', 'index', 'Left Temp Diff', 'Right Temp Diff']
    return pd.concat([d])


data1 = clean(data_train)

# 输出
# print(data1)
# print(222222222222222222222222222222222222222)

