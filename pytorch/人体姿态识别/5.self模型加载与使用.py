#单层模型使用
import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable
from torch import nn
csv_data = pd.read_csv("self_data\\one_sample.csv",sep = ',',index_col=0)
np_data=np.array([csv_data.values])
im = torch.Tensor(np_data)
if torch.cuda.is_available():
    im = Variable(im.cuda())

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

model = torch.load('model.pkl')
model.eval()
pre = model(im)    #返回值是tensor([[ 0.6357, -0.6516]], device='cuda:0', grad_fn=<CudnnBatchNormBackward>)
#print(pre)
numpy_array =pre.cpu().detach().numpy()
label_dic={0:"WALk",1:"RUN"}
predict_index=np.argmax(numpy_array, axis=1)    #横着比较，返回行号
print("predict:",label_dic[predict_index[0]])
