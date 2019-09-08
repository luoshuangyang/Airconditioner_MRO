import configparser
import os

'''读写配置文件的类'''


class ConfigFile:
    fileName = ''

    """构造函数：初始化"""
    def __init__(self, fileName):
        self.fileName = fileName
        self.cf = configparser.ConfigParser()
        self.cf.read(self.fileName, encoding="utf-8")

    """获取节为section,键值为Key的值"""
    def GetValue(self, Section, Key):
        result = self.cf.get(Section, Key)
        return result

    """获取文件路径 节为section,键值为Key的值"""
    def GetPath(self, Section, Key):
        temp = self.cf.get(Section, Key)
        # 获取当前文件路径
        # result = os.path.split(os.path.realpath(__file__))[0]
        cur_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(os.path.dirname(cur_path), 'data')
        result = data_path + temp
        return result

    """设置节为section,键值为Key的值"""
    def SetValue(self, Section, Key, Value):
        self.cf.set(Section, Key, Value)
        self.cf.write(open(self.fileName, "w"))
        return "ok"


'''
# 测试代码
configfile = os.path.join(os.getcwd(), 'Conf.ini')
cf = ConfigFile(configfile)
print(cf.GetValue("model1", "loss"))
print(cf.GetPath("operation", "Import1"))

'''
