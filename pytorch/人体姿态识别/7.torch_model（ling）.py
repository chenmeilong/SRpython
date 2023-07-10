#准确率 0.8  （3层模型）
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import  DataLoader
from torch import nn,optim
from time import time
epochs = 40
batch_size = 256
kernel_size = 3  # 卷积核的大小
pool_size = 2
dropout_rate = 0.4
n_classes = 6  # 6种行为     走路 上楼 下来 坐下 站着 躺着
f_act = 'relu'
def create_folder_try(atp_out_dir):
    """创建文件夹"""
    if not os.path.exists(atp_out_dir):
        os.makedirs(atp_out_dir)
        print ('文件夹 "%s" 不存在，创建文件夹。' % atp_out_dir)
class creat_train_data(Dataset):
    def __init__(self,data_path):
        super(creat_train_data, self).__init__()
        self.data_path=data_path
        train_path = os.path.join(self.data_path, "train")
        train_X_path = os.path.join(train_path, "Inertial Signals")

        self.X_trainS1_x = np.loadtxt(os.path.join(train_X_path, "body_acc_x_train.txt"))  # 总加速度减去重力加速度得到的数据
        self.X_trainS1_y = np.loadtxt(os.path.join(train_X_path, "body_acc_y_train.txt"))
        self.X_trainS1_z = np.loadtxt(os.path.join(train_X_path, "body_acc_z_train.txt"))
        self.X_trainS1 = np.array([self.X_trainS1_x, self.X_trainS1_y, self.X_trainS1_z])
        self.X_trainS1 = self.X_trainS1.transpose([1,0,2])  # 转置(3, 7352, 128)-----(7352, 3, 128)

        self.X_trainS2_x = np.loadtxt(os.path.join(train_X_path, "body_gyro_x_train.txt"))   # 陀螺仪角速度  （这个应该是原始数据）
        self.X_trainS2_y = np.loadtxt(os.path.join(train_X_path, "body_gyro_y_train.txt"))
        self.X_trainS2_z = np.loadtxt(os.path.join(train_X_path, "body_gyro_z_train.txt"))
        self.X_trainS2 = np.array([self.X_trainS2_x, self.X_trainS2_y, self.X_trainS2_z])
        self.X_trainS2 = self.X_trainS2.transpose([1,0,2])

        self.X_trainS3_x = np.loadtxt(os.path.join(train_X_path, "total_acc_x_train.txt"))  # 标准g测得的数据
        self.X_trainS3_y = np.loadtxt(os.path.join(train_X_path, "total_acc_y_train.txt"))
        self.X_trainS3_z = np.loadtxt(os.path.join(train_X_path, "total_acc_z_train.txt"))
        self.X_trainS3 = np.array([self.X_trainS3_x, self.X_trainS3_y, self.X_trainS3_z])
        self.X_trainS3 = self.X_trainS3.transpose([1,0,2])

        self.Y_train = np.loadtxt(os.path.join(train_path, "y_train.txt"))

        print("训练数据: ")
        print("传感器1: %s, 传感器1的X轴: %s" % (str(self.X_trainS1.shape), str(self.X_trainS1_x.shape)))
        print("传感器2: %s, 传感器2的X轴: %s" % (str(self.X_trainS2.shape), str(self.X_trainS2_x.shape)))
        print("传感器3: %s, 传感器3的X轴: %s" % (str(self.X_trainS3.shape), str(self.X_trainS3_x.shape)))
        print("传感器标签: %s" % str(self.Y_train.shape))
        print("")


    def __getitem__(self, idex):
        X_trainS1=self.X_trainS1[idex]
        X_trainS2 = self.X_trainS2[idex]
        X_trainS3 = self.X_trainS3[idex]
        Y_train   =self.Y_train[idex]
        return X_trainS1,X_trainS2,X_trainS3,Y_train
    def __len__(self):
        return len(self.Y_train)
class creat_test_data(Dataset):
    def __init__(self,data_path):
        super(creat_test_data, self).__init__()
        self.data_path=data_path
        test_path = os.path.join(self.data_path, "test")
        test_X_path = os.path.join(test_path, "Inertial Signals")

        self.X_valS1_x = np.loadtxt(os.path.join(test_X_path, "body_acc_x_test.txt"))
        self.X_valS1_y = np.loadtxt(os.path.join(test_X_path, "body_acc_y_test.txt"))
        self.X_valS1_z = np.loadtxt(os.path.join(test_X_path, "body_acc_z_test.txt"))
        self.X_valS1 = np.array([self.X_valS1_x, self.X_valS1_y, self.X_valS1_z])
        self.X_valS1 = self.X_valS1.transpose([1,0,2])

        self.X_valS2_x = np.loadtxt(os.path.join(test_X_path, "body_gyro_x_test.txt"))
        self.X_valS2_y = np.loadtxt(os.path.join(test_X_path, "body_gyro_y_test.txt"))
        self.X_valS2_z = np.loadtxt(os.path.join(test_X_path, "body_gyro_z_test.txt"))
        self.X_valS2 = np.array([self.X_valS2_x, self.X_valS2_y, self.X_valS2_z])
        self.X_valS2 = self.X_valS2.transpose([1,0,2])

        self.X_valS3_x = np.loadtxt(os.path.join(test_X_path, "total_acc_x_test.txt"))
        self.X_valS3_y = np.loadtxt(os.path.join(test_X_path, "total_acc_y_test.txt"))
        self.X_valS3_z = np.loadtxt(os.path.join(test_X_path, "total_acc_z_test.txt"))
        self.X_valS3 = np.array([self.X_valS3_x, self.X_valS3_y, self.X_valS3_z])
        self.X_valS3 = self.X_valS3.transpose([1,0,2])

        self.Y_val = np.loadtxt(os.path.join(test_path, "y_test.txt"))

        print("验证数据: ")
        print("传感器1: %s, 传感器1的X轴: %s" % (str(self.X_valS1.shape), str(self.X_valS1_x.shape)))
        print("传感器2: %s, 传感器2的X轴: %s" % (str(self.X_valS2.shape), str(self.X_valS2_x.shape)))
        print("传感器3: %s, 传感器3的X轴: %s" % (str(self.X_valS3.shape), str(self.X_valS3_x.shape)))
        print("传感器标签: %s" % str(self.Y_val.shape))
        print("\n")
    def __getitem__(self, idex):         #类的专有方法  按照索引获取值
        X_valS1=self.X_valS1[idex]
        X_valS2 = self.X_valS2[idex]
        X_valS3 = self.X_valS3[idex]
        Y_val=self.Y_val[idex]
        return X_valS1,X_valS2,X_valS3,Y_val
    def __len__(self):         #类的专有方法  获得长度
        return len(self.Y_val)

class alone_merged(nn.Module):
     def __init__(self):
         super(alone_merged, self).__init__()
         self.conv_1=nn.Conv1d(in_channels=3,out_channels=512,kernel_size=kernel_size,padding=1)
         self.batch_normalize_1=nn.BatchNorm1d(512)
         self.max_pool_1=nn.MaxPool1d(kernel_size=pool_size)
         self.drop_1=nn.Dropout(dropout_rate)
         self.conv_2=nn.Conv1d(in_channels=512,out_channels=64,kernel_size=kernel_size,padding=1)
         self.batch_normalize_2 = nn.BatchNorm1d(64)
         self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size)
         self.drop_2 = nn.Dropout(dropout_rate)
         self.conv_3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=kernel_size, padding=1)
         self.batch_normalize_3 = nn.BatchNorm1d(32)
         self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size)
         self.lstm_1=nn.LSTM(input_size=32,hidden_size=128,num_layers=1)
         self.lstm_2 =nn.LSTM(input_size=128, hidden_size=128, num_layers=1)
         self.lstm_3=nn.LSTM(input_size=128,hidden_size=128,num_layers=1)
         self.drop_3=nn.Dropout(dropout_rate)
     def forward(self,x):
         x=self.conv_1(x)#[?, 512, 128]
         x=torch.relu(x)
         x=self.batch_normalize_1(x)#[?, 512, 128]
         x=self.max_pool_1(x)#[?, 512, 64]
         x=self.drop_1(x)#[?, 512, 64]
         x=self.conv_2(x)#[?, 64, 64]
         x = torch.relu(x)
         x=self.batch_normalize_2(x)#[?, 64, 64]
         x=self.max_pool_2(x)#[?, 64, 32]
         x=self.drop_2(x)#[?, 64, 32]
         x=self.conv_3(x)#[?, 32, 32]
         x = torch.relu(x)
         x=self.batch_normalize_3(x)#[?, 32, 32]
         x=self.max_pool_3(x)#[?, 32, 16]
         x=x.permute(0,2,1)#[?, 16, 32]
         x,_=self.lstm_1(x)#[?, 16, 128]
         x,_=self.lstm_2(x)#[?, 16, 128]
         x,_=self.lstm_3(x)#[?, 16, 128]
         x=x[:,15,:]#[?,128]#只取记忆状态中的最后一个即循环序列中的最后一个
         out=self.drop_3(x)
         return out
class combine_model(nn.Module):
    def __init__(self):
        super(combine_model, self).__init__()
        self.alone_merged=alone_merged()
        self.dropout_4=nn.Dropout(dropout_rate)
        self.line=nn.Linear(384,6)
        self.beatch_normalize_4=nn.BatchNorm1d(6)
    def forward(self,x1,x2,x3):
        x1=self.alone_merged(x1)
        x2=self.alone_merged(x2)
        x3=self.alone_merged(x3)
        x=torch.cat((x1,x2,x3),dim=1)#[?,384]
        x=self.dropout_4(x)
        x=self.line(x)
        out=self.beatch_normalize_4(x)#[?,6]
        return out
train_acc_list=[]
val_acc_list=[]
def train(train_data,test_data,epochs,batch,loss_,optimizer,net,use_cuda,output_path):
    train_size=len(train_data)
    test_size=len(test_data)
    train_set=DataLoader(train_data,batch_size=batch,shuffle=True)
    test_set=DataLoader(test_data,batch_size=batch,shuffle=False)
    for epoch in range(epochs):
        since = time()
        running_loss=0
        running_acc=0
        val_loss = 0
        val_acc = 0
        for x1,x2,x3,y in train_set:
            if use_cuda:
                x1=x1.float().cuda();x2=x2.float().cuda()
                x3 = x3.float().cuda();y=y.long().cuda()
            else:
                x1 = x1.float();x2 = x2.float()
                x3 = x3.float();y = y.long()
            predict = net(x1, x2, x3)
            loss=loss_(predict,y-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            _, pred = predict.max(1)
            running_acc += (pred == y-1).sum().item()
        eplise_time = time() - since
        train_acc_list.append(running_acc / train_size)
        print('<train>LOSS:{:.6f},ACC:{:.6f},TIME:{:.0f}s'.format(running_loss / train_size,
                                                           running_acc / train_size, eplise_time))
        with torch.no_grad():
            for x1,x2,x3,y in test_set:
                if use_cuda:
                    x1 = x1.float().cuda();x2 = x2.float().cuda()
                    x3 = x3.float().cuda();y = y.long().cuda()
                else:
                    x1 = x1.float();x2 = x2.float()
                    x3 = x3.float();y = y.long()
                if x1.shape[0] != batch_size:
                    break
                predict = net(x1, x2, x3)
                loss = loss_(predict, y - 1)
                val_loss += loss.item() * y.size(0)
                _, pred = predict.max(1)
                val_acc += (pred == y-1).sum().item()
            val_acc_list.append(val_acc / test_size)
            print('<val>LOSS:{:.6f},ACC:{:.6f}'.format(val_loss / test_size,
                                                       val_acc / test_size))

    model_path = os.path.join(output_path, "merged_dcl.h5")
    torch.save(net.state_dict(),model_path)
import matplotlib.pyplot as plt
def show_train_history(train_acc_list, val_acc_list):
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__=="__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset")  # 数据集路径
    output = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset_output")  # 输出路径
    create_folder_try(output)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net=combine_model().cuda()
    else:
        net=combine_model()
    loss=torch.nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(),lr=0.01, alpha=0.9)
    #optimizer = torch.optim.SGD(net.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
    train_data=creat_train_data(data)
    test_data=creat_test_data(data)
    train(train_data,test_data,epochs,batch_size,loss,optimizer,net,use_cuda,output)
    show_train_history(train_acc_list,val_acc_list)

