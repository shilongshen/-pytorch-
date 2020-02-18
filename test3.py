"""
        利用torch训练神经网络举例，参考https://pytorch.org/tutorials/beginner/nn_tutorial.html
        使用mnist数据集进行训练，其图片为0-9数字灰度图，每行代表一张图片，训练集维度为50000*784
        网络只包含了线性层和一个激活层
"""
from pathlib import Path
import requests
import pickle
import gzip
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import torch
import math

"""
        导入mnist数据集
"""
FILENAME = "mnist.pkl.gz"
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


#plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
#plt.show()
#print(x_train.shape)
#print(y_train)
#print((y_train.shape))

#将numpy转换为tensor
x_train,y_train,x_valid,y_valid=map(torch.tensor,(x_train,y_train,x_valid,y_valid))
n,c=x_train.shape
#print(x_train,y_train)
#print(x_train.shape)

#初始化权重
weights=torch.randn(784,10)/math.sqrt(784)
#print(weights.shape)
weights.requires_grad_()
bias=torch.zeros(10,requires_grad=True)



"""
        定义前向传播
"""

#激活函数使用log_softmax函数
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
        return log_softmax(xb@weights+bias)

bs=64#bitch size
xb=x_train[0:bs]
#print(xb.shape)
preds=model(xb)
#preds=model(preds)

"""
        定义negative log_likehood 损失函数(负对数似然损失函数函数）
"""
def nll(input,targrt):
        return -input[range(targrt.shape[0]),targrt].mean()
loss_func=nll

yb = y_train[0:bs]
print('未训练前的损失函数值：',loss_func(preds,yb))

"""
        定义计算准确度的函数
        在每次预测中，输出向量最大值得下标索引如果和目标值（标签）相同，则认为预测结果是对的。
"""
def accuracy(out,yb):
        preds=torch.argmax(out,dim=1)#返回每行中最大元素所在列的位置
        return (preds==yb).float().mean()

print('未训练前的准确率：',accuracy(preds,yb))
"""
        开始循环训练模型
        1.选择一个mini bitch
        2.用模型得到预测结果
        3.计算损失函数值
        4.loss.backward()更新模型的梯度，在这里是权值（weights）和偏移（bias）
        5.用梯度去更新权值和偏移
"""
from IPython.core.debugger import set_trace
lr=0.9#学习率
epochs=10
for epochs in range(epochs):
        for i in range((n-1)//bs+1):
                start_i=i*bs
                end_i=start_i+bs
                xb=x_train[start_i:end_i]
                yb=y_train[start_i:end_i]
                preds=model(xb)
                loss=loss_func(preds,yb)
                loss.backward()#自动求导
                """
                        将梯度更新语句包在 with torch.no_grad(): 之下 的原因
                        由于这个阶段的optimizer是你自己写的SGD，
                        只需要进行数值计算，不需要创建计算图（默认PyTorch会给张量计算创建计算图），
                        所以关掉这个功能，用no_grad这个上下文管理器，在作用域内只做计算，不记录计算图。
                """
                with torch.no_grad():
                        weights-=weights.grad*lr
                        #weights=weights-weights.grad*lr
                        #bias=bias-bias.grad*lr
                        bias-=bias.grad*lr
                        """
                                weights.grad.zero_()
                                bias.grad.zero_()--梯度清零的原因：
                                根据pytorch中的backward()函数的计算，当网络参量进行反馈时，
                                梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将
                                两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。
                        """
                        weights.grad.zero_()
                        bias.grad.zero_()

print('训练后的损失函数：',loss_func(model(xb),yb))
print('训练后的准确率：',accuracy(model(xb),yb))



