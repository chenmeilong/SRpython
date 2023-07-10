# -- coding: utf-8 --
#训练数据:  传感器1: (7352, 128, 3), 传感器1的X轴: (7352, 128)     一共7352条数据
#
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

def create_folder_try(atp_out_dir):
    """创建文件夹"""
    if not os.path.exists(atp_out_dir):
        os.makedirs(atp_out_dir)
        print ('文件夹 "%s" 不存在，创建文件夹。' % atp_out_dir)

def load_data(data_path):
    """
    加载本地的UCI的训练数据和验证数据
    :param data_path 数据集
    :return: 训练数据和验证数据
    """
    train_path = os.path.join(data_path, "train")
    train_X_path = os.path.join(train_path, "Inertial Signals")

    X_trainS1_x = np.loadtxt(os.path.join(train_X_path, "body_acc_x_train.txt"))    #总加速度减去重力加速度得到的数据
    X_trainS1_y = np.loadtxt(os.path.join(train_X_path, "body_acc_y_train.txt"))
    X_trainS1_z = np.loadtxt(os.path.join(train_X_path, "body_acc_z_train.txt"))
    X_trainS1 = np.array([X_trainS1_x, X_trainS1_y, X_trainS1_z])
    X_trainS1 = X_trainS1.transpose([1, 2, 0])                                         #转置(3, 7352, 128)-----(7352, 128, 3)

    X_trainS2_x = np.loadtxt(os.path.join(train_X_path, "body_gyro_x_train.txt"))   #陀螺仪角速度  （这个应该是原始数据）
    X_trainS2_y = np.loadtxt(os.path.join(train_X_path, "body_gyro_y_train.txt"))
    X_trainS2_z = np.loadtxt(os.path.join(train_X_path, "body_gyro_z_train.txt"))
    X_trainS2 = np.array([X_trainS2_x, X_trainS2_y, X_trainS2_z])
    X_trainS2 = X_trainS2.transpose([1, 2, 0])

    X_trainS3_x = np.loadtxt(os.path.join(train_X_path, "total_acc_x_train.txt"))    # 标准g测得的数据
    X_trainS3_y = np.loadtxt(os.path.join(train_X_path, "total_acc_y_train.txt"))
    X_trainS3_z = np.loadtxt(os.path.join(train_X_path, "total_acc_z_train.txt"))
    X_trainS3 = np.array([X_trainS3_x, X_trainS3_y, X_trainS3_z])
    X_trainS3 = X_trainS3.transpose([1, 2, 0])

    Y_train = np.loadtxt(os.path.join(train_path, "y_train.txt"),dtype=np.int32)
    Y_train = Y_train - 1    # 标签是从1开始   转换为one-hot编码

    # print ("训练数据: ")
    # print ("传感器1: %s, 传感器1的X轴: %s" % (str(X_trainS1.shape), str(X_trainS1_x.shape)))
    # print ("传感器2: %s, 传感器2的X轴: %s" % (str(X_trainS2.shape), str(X_trainS2_x.shape)))
    # print ("传感器3: %s, 传感器3的X轴: %s" % (str(X_trainS3.shape), str(X_trainS3_x.shape)))
    # print ("传感器标签: %s" % str(Y_train.shape))
    # print (Y_train)
    # print ("")

    test_path = os.path.join(data_path, "test")
    test_X_path = os.path.join(test_path, "Inertial Signals")

    X_valS1_x = np.loadtxt(os.path.join(test_X_path, "body_acc_x_test.txt"))
    X_valS1_y = np.loadtxt(os.path.join(test_X_path, "body_acc_y_test.txt"))
    X_valS1_z = np.loadtxt(os.path.join(test_X_path, "body_acc_z_test.txt"))
    X_valS1 = np.array([X_valS1_x, X_valS1_y, X_valS1_z])
    X_valS1 = X_valS1.transpose([1, 2, 0])

    X_valS2_x = np.loadtxt(os.path.join(test_X_path, "body_gyro_x_test.txt"))
    X_valS2_y = np.loadtxt(os.path.join(test_X_path, "body_gyro_y_test.txt"))
    X_valS2_z = np.loadtxt(os.path.join(test_X_path, "body_gyro_z_test.txt"))
    X_valS2 = np.array([X_valS2_x, X_valS2_y, X_valS2_z])
    X_valS2 = X_valS2.transpose([1, 2, 0])

    X_valS3_x = np.loadtxt(os.path.join(test_X_path, "total_acc_x_test.txt"))
    X_valS3_y = np.loadtxt(os.path.join(test_X_path, "total_acc_y_test.txt"))
    X_valS3_z = np.loadtxt(os.path.join(test_X_path, "total_acc_z_test.txt"))
    X_valS3 = np.array([X_valS3_x, X_valS3_y, X_valS3_z])
    X_valS3 = X_valS3.transpose([1, 2, 0])

    Y_val = np.loadtxt(os.path.join(test_path, "y_test.txt"),dtype=np.int32)
    Y_val = Y_val - 1

    # print ("验证数据: ")
    # print ("传感器1: %s, 传感器1的X轴: %s" % (str(X_valS1.shape), str(X_valS1.shape)))
    # print ("传感器2: %s, 传感器2的X轴: %s" % (str(X_valS2.shape), str(X_valS2.shape)))
    # print ("传感器3: %s, 传感器3的X轴: %s" % (str(X_valS3.shape), str(X_valS3.shape)))
    # print ("传感器标签: %s" % str(Y_val.shape))
    # print ("\n")
    print("数据加载成功")

    return X_trainS1, X_trainS2, X_trainS3, Y_train, X_valS1, X_valS2, X_valS3, Y_val    #3个训练数据+1个训练标签    3个测试数据+一个测试标签


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return float(num_correct)/float(total)

def train(net, train_data,train_tabel,test_data,test_tabel, num_epochs,batch_size,optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    train_total_batch=int(len(train_data[0]) / batch_size)         #迭代一次训练多少批次28      7352/256
    test_total_batch=int(len(test_data[0]) / batch_size)          #11
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for i in range(train_total_batch):
            #print(train_data.shape)         #(3, 7352, 128, 3)
            im=train_data[:,batch_size * i:batch_size * i + batch_size,:,:]
            #print(im.shape)                  #(3, 256, 128, 3)
            label=train_tabel[batch_size *i:batch_size *i+batch_size]
            #print(train_tabel.shape)        #(7352,)
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
        for i in range(test_total_batch):         #11
            im = test_data[:,batch_size * i:batch_size * i + batch_size,:,:]
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

def main(data_path, output_path):
    X_trainS1, X_trainS2, X_trainS3, Y_train, X_valS1, X_valS2, X_valS3, Y_val = load_data(data_path)
    # 总加速度减去重力加速度得到的数据，陀螺仪角速度数据，标准g测得的数据，训练的标签，总加速度减去重力加速度得到的数据，陀螺仪角速度数据，标准g测得的数据，测试标签
    epochs = 50         #迭代次数
    batch_size = 256     #一个批次数据
    kernel_size = 3         #卷积核的大小
    pool_size = 2
    dropout_rate = 0.5
    n_classes = 6        #6种行为     走路 上楼 下楼 坐下 站着 躺着
    train_data=np.asarray([X_trainS1,X_trainS2, X_trainS3])
    train_val=Y_train
    test_data=np.asarray([X_valS1, X_valS2, X_valS3])
    test_val= Y_val
    # 单层模型
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
                nn.BatchNorm1d(32),                                # 归一化
                nn.ReLU(True),
                nn.MaxPool1d(pool_size, 2),
            )
            self.one_lstm=torch.nn.LSTM(input_size=32,hidden_size=128,num_layers=1)
            self.two_lstm=torch.nn.LSTM(input_size=128,hidden_size=128,num_layers=1)
            self.three_lstm=torch.nn.LSTM(input_size=128,hidden_size=128,num_layers=1)
            self.dropout=torch.nn.Dropout(dropout_rate)
            self.classfy = torch.nn.Linear(384, n_classes)
            self.Batch_Norma =torch.nn.BatchNorm1d(6)    #来自期望输入的特征数
        def forward(self, data):            # (3, 256, 128, 3)
            one_data = data[0].squeeze()
            one_data = one_data.permute(0, 2, 1)  # 将第三维度换到第二维度上 即[256,3,128]
            con_pool_out = self.con_pool(one_data)
            con_pool_out = con_pool_out.squeeze()
            con_pool_out = con_pool_out.permute(0, 2, 1)  # 将第三维度换到第二维度上 即[256,3,128]
            one_lstm_out, _ = self.one_lstm(con_pool_out)
            two_lstm_out, _ = self.two_lstm(one_lstm_out)
            three_lstm_out, _ = self.three_lstm(two_lstm_out)
            one_cnn_lstm_out=self.dropout(three_lstm_out[:,-1,:])

            two_data = data[1].squeeze()
            two_data = two_data.permute(0, 2, 1)  # 将第三维度换到第二维度上 即[256,3,128]
            con_pool_out = self.con_pool(two_data)
            con_pool_out = con_pool_out.squeeze()
            con_pool_out = con_pool_out.permute(0, 2, 1)  # 将第三维度换到第二维度上 即[256,3,128]
            one_lstm_out, _ = self.one_lstm(con_pool_out)
            two_lstm_out, _ = self.two_lstm(one_lstm_out)
            three_lstm_out, _ = self.three_lstm(two_lstm_out)
            two_cnn_lstm_out=self.dropout(three_lstm_out[:,-1,:])

            three_data = data[2].squeeze()
            three_data = three_data.permute(0, 2, 1)  # 将第三维度换到第二维度上 即[256,3,128]
            con_pool_out = self.con_pool(three_data)
            con_pool_out = con_pool_out.squeeze()
            con_pool_out = con_pool_out.permute(0, 2, 1)  # 将第三维度换到第二维度上 即[256,3,128]
            one_lstm_out, _ = self.one_lstm(con_pool_out)
            two_lstm_out, _ = self.two_lstm(one_lstm_out)
            three_lstm_out, _ = self.three_lstm(two_lstm_out)
            three_cnn_lstm_out=self.dropout(three_lstm_out[:,-1,:])

            cnn_lstm_cell_out=torch.cat([one_cnn_lstm_out, two_cnn_lstm_out,three_cnn_lstm_out], 1)
            dropout_out=self.dropout(cnn_lstm_cell_out)
            classfy_out=self.classfy(dropout_out)
            out=self.Batch_Norma(classfy_out)
            return out

    criterion = nn.CrossEntropyLoss()
    net = cnn_lstm_cell()
    # print(net) #查看模型
    optimizer = torch.optim.SGD(net.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
    train(net,train_data,train_val,test_data,test_val, epochs,batch_size, optimizer, criterion)
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
    test_total_batch=int(len(test_data[0]) / batch_size)
    valid_loss = 0
    valid_acc = 0
    for i in range(test_total_batch):
        im = test_data[:,batch_size * i:batch_size * i + batch_size,:,:]
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
    torch.save(net.state_dict(), output_path+'/params.pkl')   # 仅保存和加载模型参数(推荐使用)
    # torch.save(net, output_path+'/model.pkl')                # 保存和加载整个模型

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))                     # 当前目录
    data = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset")              #数据集路径
    output = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset_output")    #输出路径
    create_folder_try(output)                                                #检测文件夹是否存在 不存在创建
    main(data_path=data, output_path=output)
