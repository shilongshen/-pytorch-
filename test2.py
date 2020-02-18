import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import math

"""
    文件使用完毕后必须关闭，因为文件对象会占用操作系统的资源,
    Python引入了with语句来自动帮我们调用close()方法
    该部分代码为读取文件
"""
with open('data.txt','r') as f:
    data=f.readlines()#调用readline()可以每次读取一行内容，调用readlines()一次读取所有内容并按行返回list
    data = [row.split(',') for row in data][:-1]
    x_train=[(float(i[0]),float(i[1])) for i in data]
    y_train=[float(i[2]) for i in data]
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_train=torch.from_numpy(x_train)
    y_train=torch.from_numpy(y_train)
    x_train=x_train.float()#将数据类型转换为float
    y_train=y_train.float()
    y_train=y_train.unsqueeze(1)#将y_train的大小由[99],变为[99,1]
"""
    该部分代码为画图
"""
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
    定义logistic回归模型，即定义一个计算图，只需要定义forward方法，后向函数（自动计算微分的函数）就会自动定义
"""
class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
        self.lr=nn.Linear(2,1)#定义线性层
        self.sm=nn.Sigmoid()#定义激活层（非线性层）
    def forward(self, x):#设定前向传播的步骤
        x=self.lr(x)#z=xw+b
        x=self.sm(x)#a=sigmoid(z)
        return x

net=mynet()
critetion=nn.BCELoss()#定义二分类损失函数
optimizer=torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)#随机梯度下降优化函数
epochs=50000
for epoch in range(epochs):
    out=net(x_train)
    loss=critetion(out,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

w,b=net.parameters()
print(w,b)
w0=w[0,0].item()
w1=w[0,1].item()
b=b.item()

import numpy as np
plot_x=np.arange(30,100,0.1)
plot_y=(-w0*plot_x-b)/w1
plt.plot(plot_x,plot_y)
plt.show()

#w0=w0.numpy()
#print(w0)

"""

for epoch in range(50000):
    x=Variable(x_data)
    y=Variable(y_data)

#前向传播
out=net(x)
loss=critetion(out,y)
print_loss=loss.data[0]
mask=out.ge(0.5).float()
correct=(mask==y),sum()
acc=correct.data[0]/x.size(0)

#反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()
if (epoch+1)%1000==0:
    print('*'*10)
    print('epoch{}'.format(epoch+1))
    print('loss is {:.4f}'.format(print_loss))
    print('acc is {:4f}'.format(acc))
"""