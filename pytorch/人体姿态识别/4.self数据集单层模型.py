# -- coding: utf-8 --

import os
import torch
import numpy as np
from torch import nn
from datetime import datetime
from torch.autograd import Variable

plt_train_loss=[]
plt_test_loss=[]
plt_train_acc=[]
plt_test_acc=[]

def load_data(data_path):
    train_X_path = os.path.join(data_path, "train")

    X_trainS1_x = np.loadtxt(os.path.join(train_X_path, "acc_train_x.txt"))    #acc加速度数据
    X_trainS1_y = np.loadtxt(os.path.join(train_X_path, "acc_train_y.txt"))
    X_trainS1_z = np.loadtxt(os.path.join(train_X_path, "acc_train_z.txt"))
    X_trainS1 = np.array([X_trainS1_x, X_trainS1_y, X_trainS1_z])
    X_trainS1 = X_trainS1.transpose([1, 2, 0])                                         #转置(3, 7352, 128)-----(7352, 128, 3)


    Y_train = np.loadtxt(os.path.join(train_X_path, "acc_train_label.txt"),dtype=np.int32)
    Y_train = Y_train - 1

    print ("训练数据: ")
    print ("传感器acc: %s, 传感器acc的X轴: %s" % (str(X_trainS1.shape), str(X_trainS1_x.shape)))
    print ("传感器标签: %s" % str(Y_train.shape))
    print ("")

    test_X_path= os.path.join(data_path, "test")

    X_valS1_x = np.loadtxt(os.path.join(test_X_path, "acc_test_x.txt"))
    X_valS1_y = np.loadtxt(os.path.join(test_X_path, "acc_test_y.txt"))
    X_valS1_z = np.loadtxt(os.path.join(test_X_path, "acc_test_z.txt"))
    X_valS1 = np.array([X_valS1_x, X_valS1_y, X_valS1_z])
    X_valS1 = X_valS1.transpose([1, 2, 0])

    Y_val = np.loadtxt(os.path.join(test_X_path, "acc_test_label.txt"),dtype=np.int32)
    Y_val =  Y_val - 1

    print ("验证数据: ")
    print ("传感器acc: %s, 传感器acc的X轴: %s" % (str(X_valS1.shape), str(X_valS1.shape)))
    print ("传感器标签: %s" % str(Y_val.shape))
    print ("\n")

    return X_trainS1, Y_train, X_valS1, Y_val    #


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return float(num_correct)/float(total)

def train(net, train_data,train_tabel,test_data,test_tabel, num_epochs,batch_size,optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    train_total_batch=int(len(train_data) / batch_size)         #迭代一次训练多少批次28
    test_total_batch=int(len(test_data) / batch_size)
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for i in range(train_total_batch):
            im=train_data[batch_size * i:batch_size * i + batch_size]
            label=train_tabel[batch_size *i:batch_size *i+batch_size]
            #print(np.shape(im))
            im = torch.Tensor(im)
            label=label.tolist()
            label = torch.tensor(label)
            if torch.cuda.is_available():
                im = Variable(im.cuda())
                label = Variable(label.cuda())
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output,label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        valid_loss = 0
        valid_acc = 0
        net = net.eval()
        for i in range(test_total_batch):
            im = test_data[batch_size * i:batch_size * i + batch_size]
            label = test_tabel[batch_size * i:batch_size * i + batch_size]
            im = torch.Tensor(im)
            label = label.tolist()
            label = torch.tensor(label)
            if torch.cuda.is_available():
                im = Variable(im.cuda())
                label = Variable(label.cuda())
            else:
                im = Variable(im)
                label = Variable(label)
            output = net(im)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += get_acc(output, label)
        epoch_str = (
            "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
            % (epoch, train_loss / train_total_batch,
               train_acc / train_total_batch, valid_loss / test_total_batch,
               valid_acc / test_total_batch))
        prev_time = cur_time
        print(epoch_str + time_str)
        plt_train_loss.append(train_loss / train_total_batch)
        plt_test_loss.append(valid_loss / test_total_batch)
        plt_train_acc.append(train_acc / train_total_batch)
        plt_test_acc.append(valid_acc / test_total_batch)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))     # 当前目录  D:\Wayne\Desktop\behavior recognition\code\newtrain
data_path = os.path.join(ROOT_DIR, "self_data")              #数据集路径
output_path = os.path.join(ROOT_DIR, "self_data")    #输出路径
#main(data_path=data, output_path=output)

# 定义模型
kernel_size = 3         #卷积核的大小
pool_size = 2
dropout_rate = 0.4
n_classes =2        #6种行为     走路 上楼 下楼 坐下 站着 躺着
class cnn_lstm_cell(nn.Module):
    def __init__(self):
        super(cnn_lstm_cell, self).__init__()
        self.con_pool = nn.Sequential(
            nn.Conv1d(3, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),           # 归一化
            nn.ReLU(True),
            nn.MaxPool1d(pool_size, 2),                         #池化 大小  和步长
            nn.Dropout(dropout_rate),
            nn.Conv1d(512, 64, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(64),  # 归一化
            nn.ReLU(True),
            nn.MaxPool1d(pool_size, 2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 32,kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(32),  # 归一化
            nn.ReLU(True),
            nn.MaxPool1d(pool_size, 2),
        )
        self.one_lstm=torch.nn.LSTM(input_size=32,hidden_size=128,num_layers=1)
        self.two_lstm=torch.nn.LSTM(input_size=128,hidden_size=128,num_layers=1)
        self.three_lstm=torch.nn.LSTM(input_size=128,hidden_size=128,num_layers=1)
        self.dropout=torch.nn.Dropout(dropout_rate)
        self.classfy = torch.nn.Linear(128, n_classes)
        self.Batch_Norma   =torch.nn.BatchNorm1d(n_classes)    #来自期望输入的特征数  一般取100

    def forward(self, x):
        #x = x.squeeze()
        x = x.permute(0, 2, 1)  # 将第三维度换到第二维度上 即[256,3,128]
        con_pool_out = self.con_pool(x)
        con_pool_out = con_pool_out.permute(0, 2, 1)  # 将第三维度换到第二维度上 即[256,3,128]
        one_lstm_out, _ = self.one_lstm(con_pool_out)
        two_lstm_out, _ = self.two_lstm(one_lstm_out)
        three_lstm_out, _ = self.three_lstm(two_lstm_out)
        dropout_out=self.dropout(three_lstm_out[:,-1,:])
        classfy_out=self.classfy(dropout_out)
        out=self.Batch_Norma(classfy_out)
        return out


X_trainS1, Y_train, X_valS1, Y_val = load_data(data_path)
epochs = 50         #迭代次数
batch_size = 256     #一个批次数据

train_data=X_trainS1
train_val=Y_train
test_data= X_valS1
test_val= Y_val

net = cnn_lstm_cell()
criterion = nn.CrossEntropyLoss()
# print(net) #查看模型
optimizer = torch.optim.SGD(net.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
train(net,train_data,train_val,test_data,test_val, epochs,batch_size, optimizer, criterion)
#torch.save(net.state_dict(), output_path+'\\params.pkl')   # 仅保存和加载模型参数(推荐使用)
torch.save(net,"model.pkl")                # 保存和加载整个模型
#画图
import matplotlib.pyplot as plt
def show_train_history(train_acc, test_acc):
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
show_train_history(plt_train_acc,plt_test_acc)
show_train_history(plt_train_loss,plt_test_loss)

#测试
test_total_batch=int(len(test_data) / batch_size)
valid_loss = 0
valid_acc = 0
for i in range(test_total_batch):
    im = test_data[batch_size * i:batch_size * i + batch_size]
    label = test_val[batch_size * i:batch_size * i + batch_size]
    im = torch.Tensor(im)
    label = label.tolist()
    label = torch.tensor(label)
    if torch.cuda.is_available():
        im = Variable(im.cuda())
        label = Variable(label.cuda())
    else:
        im = Variable(im)
        label = Variable(label)
    output = net(im)
    loss = criterion(output, label)
    valid_loss += loss.item()
    valid_acc += get_acc(output, label)
print(valid_acc/test_total_batch)


