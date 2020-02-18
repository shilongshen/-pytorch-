"""
        利用torch训练神经网络举例，参考https://pytorch.org/tutorials/beginner/nn_tutorial.html
        使用mnist数据集进行训练，其图片为0-9数字灰度图，每行代表一张图片，训练集维度为50000*784
        网络只包含了线性层和一个激活层
        引入 nn.Module和nn.Parameter 改进代码
        引入nn.Linear来定义线性层，而不用手动设置权重和参数
        使用torch.optim改进训练过程，而不用手动更新参数
        引入Dataset处理数据,TensorDataset 是一个包含张量的数据集。通过定义长度索引等方式，使我们更好地利用索引，切片等方法迭代数据。这会让我们很容易地在一行代码中获取我们地数据。
        DataLoader 用于批量加载数据，你可以用他来加载任何来自 Dataset的数据，它使得数据的批量加载十分容易。
        添加测试集
        创建 fit() 和 get_data() 优化代码
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
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
"""
        导入mnist数据集
"""
FILENAME = "mnist.pkl.gz"
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# plt.show()
# print(x_train.shape)
# print(y_train)
# print((y_train.shape))

# 将numpy转换为tensor
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape
bs = 64  # bitch size
xb = x_train[0:bs]
yb = y_train[0:bs]
train_ds=TensorDataset(x_train,y_train)#x_train y_train可以被组合进一个TensorDataset中，这会使得迭代切片更加简单。
#train_dl=DataLoader(train_ds,batch_size=bs,shuffle=True)#训练数据的批量加载，即将数据按照bs大小为一份进行分割.shuffle() 方法将序列的所有元素随机排序，打乱数据的分布有助于减小每一批(batch)数据间的关联，有利于模型的泛化。
valid_ds=TensorDataset(x_valid,y_valid)
#valid_dl=DataLoader(valid_ds,batch_size=bs*2)#测试数据的批量加载
#weights = torch.randn(784, 10) / math.sqrt(784)
#weights.requires_grad_()
#bias = torch.zeros(10, requires_grad=True)

"""
    定义激活函数和损失函数
    如果我们的网络中使用 negative log likelihood loss 作为损失函数， 
    log softmax activation 作为激活函数 （即我们上面实现的损失函数与激活函数）。
    在pytorch中我们直接使用函数 F.cross_entropy 便可实现上面两个函数的功能
"""
import torch.nn.functional as F

loss_func = F.cross_entropy



"""
    定义计算准确度的函数
"""


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)  # 返回每行中最大元素所在列的位置
    return (preds == yb).float().mean()


"""
    定义网络模型
"""


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)  # 此处用nn.Linear来代替手动定义 权重 与 偏置

    def forward(self, xb):
        return self.lin(xb)  # 计算xb@self.weights+self.bias


model = Mnist_Logistic()
print('未训练前的损失函数', loss_func(model(xb), yb))  # 注意此处的model(xb)表示的是线性层的输出，而test3中的表示激活层的输出
#print('未训练前的准确率', accuracy(model(xb), yb))#此处不能再用此公式计算准确率了

"""
        开始循环训练模型
        1.选择一个mini bitch
        2.用模型得到预测结果
        3.计算损失函数值
        4.loss.backward()更新模型的梯度，在这里是权值（weights）和偏移（bias）
        5.用梯度去更新权值和偏移
"""
lr = 0.9  # 学习率
epochs = 5
# 使用torch.optim改进训练过程，而不用手动更新参数
from torch import optim


def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)


model, opt = get_model()

"""
    为训练集添加优化器，并执行反向传播，对于测试集我们不添加优化器，当然也不会执行反向传播
"""
def loss_batch(model,loss_func,xb,yb,opt=None):
    loss=loss_func(model(xb),yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(),len(xb)#loss.item返回loss中的元素

import numpy as np
def fit(epochs,model,loss_func,opt,train_dl,vaild_dl):
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
           loss_batch(model,loss_func,xb,yb,opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        #with torch.no_grad:
         #   losses,nums=zip(*[loss_batch(model,loss_func,xb,yb,opt) for xb,yb in vaild_dl])
          #  val_loss=np.sum(np.multiply(losses,nums))/np.sum(nums)
        print(epoch,val_loss)

def get_data(train_ds,valid_ds,bs):
    return (DataLoader(train_ds,batch_size=bs,shuffle=True),DataLoader(valid_ds,batch_size=bs*2))


"""
    现在，获取数据加载模型进行训练的整个过程只需要三行代码便能实现了
"""
train_dl,valid_dl=get_data(train_ds,valid_ds,bs)
model,opt=get_model()
fit(epochs,model,loss_func,opt,train_dl,valid_dl)
"""
#使用DataLoader加载数据
for epoch in range(epochs):
    model.train()#进入训练模式
    for xb,yb in train_dl:
        pred=model(xb)
        loss=loss_func(pred,yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()#进入测试模式
    with torch.no_grad():
        valid_loss=sum(loss_func(model(xb),yb) for xb,yb in valid_dl)
    print('第 %d 个epoch循环测试的准确度'%epoch,valid_loss/len(valid_dl))
    print('第 %d 个epoch循环训练的准确度'%epoch,accuracy(model(xb),yb))
    print('################################################################')
"""
"""
for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        #xb = x_train[start_i:end_i]
        #yb = y_train[start_i:end_i]
        xb,yb=train_ds[start_i:end_i]#使用Dataset加载数据
        loss = loss_func(model(xb), yb)

        loss.backward()  # 计算梯度，
        opt.step()  # 更新参数
        opt.zero_grad()  # 梯度归零
"""
"""
def fit():
    for epoch in range(epochs):
        for i in range((n-1)//bs+1):
            start_i=i*bs
            end_i=start_i+bs
            xb=x_train[start_i:end_i]
            yb=y_train[start_i:end_i]
            pred=model(xb)
            loss=loss_func(pred,yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p-=p.grad*lr
                model.zero_grad()

fit()
"""
print('训练后的损失函数', loss_func(model(xb), yb))
#print('训练后的准确率', accuracy(model(xb), yb))

###########################################################
# 这是我的练习代码

####################################################