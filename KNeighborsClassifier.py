#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#facebook预测入住案例：K近邻算法

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#一、读取数据
data = pd.read_csv('C:/Study/exe/facebook/train.csv')
# print(data.head(10))

#二、处理数据
#1.缩小数据（只是练习，方便教学）查询数据筛选 query() 在dataframe里面使用
data = data.query("x>1.0 & x<2.0 & y>2.5 & y<3.5")   #选取小范围数据

#2.处理时间
time_values = pd.to_datetime(data['time'],unit='s')
# print(time_values)

#日期转化为字典格式
time_values = pd.DatetimeIndex(time_values)

#构造一些特征，day,month
data['day'] = time_values.day
data['hour'] = time_values.hour
data['weekday'] = time_values.weekday

#把时间戳特征删除
data = data.drop('time',axis=1)
# print(data)
# data.info()

#把签到数量少于n个目标位置删除  groupby(),reset_index()
place_count = data.groupby('place_id').count()
tf = place_count[place_count.row_id>3].reset_index()
data = data[data['place_id'].isin(tf.place_id)]
print(place_count)
print(tf)
print(data)

#三、取出数据中的特征值和目标值 并且删除row_id,减少影响
y =data['place_id']
x = data.drop(['place_id','row_id'],axis=1)
print(x)

#进行数据分割训练集，测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

#四、特征工程(标准化) 最后做这步 与不标准化进行对比
std = StandardScaler()
##对测试和训练集均进行标准化
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)  #前面fit_transform以及fit算出平均值和标准差，就不需要再fit,涉及转换器和预估器知识

#五、进行算法流程，模型拟合
kn = KNeighborsClassifier(n_neighbors=5)

#1.fit,predict,score
kn.fit(x_train,y_train)

#2.得出预测结果
y_predict = kn.predict(x_test)
print("预测目标签到位置为：",y_predict)

#3.得出准确率
score = kn.score(x_test,y_test)
print("预测的准确率：",score)
