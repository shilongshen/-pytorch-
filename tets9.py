import numpy as np
import torch
import matplotlib.pyplot as plt

with open('data.txt','r') as f:
    data_list=f.readlines()

#print('未转换前的类型',type(data))
#print(data)
data_list=[i.split('\n')[0] for i in data_list]
data_list = [row.split(',') for row in data_list][:-1]
data_list=[(float(i[0]),float(i[1]),float(i[2])) for i in data_list]
data_list=np.array(data_list)
data_list=torch.from_numpy(data_list)
#print(data)

x0=list(filter(lambda x:x[-1]==0.0,data_list))#筛选出每一行最后一个元素为0的行
x1=list(filter(lambda x:x[-1]==1.0,data_list))#筛选出每一行最后一个元素为1的行
plot_x0_0=[i[0] for i in x0]
plot_x0_1=[i[1] for i in x0]
plot_x1_0=[i[0] for i in x1]
plot_x1_1=[i[1] for i in x1]
plt.plot(plot_x0_0,plot_x0_1,'ro',label='x_0')
plt.plot(plot_x1_0,plot_x1_1,'bo',label='x_1')
plt.legend(loc='best')
plt.show()


"""
print(type(data))
print(data)
x_train=[(float(i[0]),float(i[1])) for i in data]
y_train=[(float(i[2])) for i in data]
y_train=np.array(y_train)
y_train=torch.from_numpy(y_train)

print('y_train',y_train)
print(x_train)
print(type(x_train))
x_train=np.array(x_train)
print(x_train)
print(type(x_train))
x_train=torch.from_numpy(x_train)
print(type(x_train))

import torch
import numpy as np

#print(data)
#data=[i.split('\n') for i in data]
"""
"""
mylist = [[1, 2, 3], [4, 5, 6]]  # 列表
print(type(mylist))
print(mylist, end='\n\n')
myarray = np.array(mylist)  # 列表转数组
print(type(myarray))
print(myarray, end="\n\n")
"""