#!/usr/bin/env python
# coding: utf-8

# ## Char RNN 生成文本
# 在循环神经网络的章节我们了解到其非常擅长处理序列问题，那么对于文本而言，其也相当于一个序列，因为每句话都是由单词或者汉子按照序列顺序组成的，所以我们也能够使用 RNN 对其进行处理，那么如何能够生成文本呢？其实原理非常简单，下面我们来讲一讲 Char RNN。
# 
# ### 训练过程
# 前面我们介绍过 RNN 的输入和输出存在多种关系，比如一对多，多对多等等，不同的输入对应着不同的应用，比如多对多可以用来做机器翻译等等，今天我们要讲的 Char RNN 在训练网络的时候是一个相同长度的多对多类型，也就是输入一个序列，输出一个吸纳共同长度的序列。
# 
# 具体的网络训练过程如下
# 
# <img src=https://ws1.sinaimg.cn/large/006tNc79gy1fob5kq3r8jj30mt09dq2r.jpg width=700>

# 可以看到上面的网络流程中，输入是一个序列 "床 前 明 月 光"，输出也是一个序列 "前 明 月 光 床"。如果你仔细观察可以发现网络的每一步输出都是下一步的输入，这是不是某种巧合呢？
# 
# 并不是的，这就是 Char RNN 的设计思路，对于任意的一句话，比如 "我喜欢小猫"，我们可以将其拆分为 Char RNN 的训练集，输入就是 "我 喜 欢 小 猫"，这构成了长度为 5 的序列，网络的每一步输出就是 "喜 欢 小 猫 我"。当然对于一个序列，其最后一个字符后面并没有其他的字符，所以有多种方式选择，比如将序列的第一个字符作为其输出，也就是 "光" 的输出是 "床"，或者将其本身作为输出，也就是 "光" 的输出是 "光"。
# 
# 这样设计有什么好处呢？因为训练的过程是一个监督的训练的过程，所以并不能看出这样做的意义，在生成文本的过程我们就能够看出这样做到底有什么好处。
# 
# ### 生成文本
# 我们直接讲解一下生成文本的过程，就能够直观的解释训练过程的原因。
# 
# 首先需要输入网络一段初始的序列进行预热，预热的过程并不需要实际的输出结果，只是为了生成拥有记忆效果的隐藏状态，并将隐藏状态保留下来，接着我们开始正式生成文本，不断地生成新的句子，这个过程是可以无限循环下去，或者到达我们的要求输出长度，具体可以看看下面的图示
# 
# <img src=https://ws2.sinaimg.cn/large/006tNc79gy1fob5z06w1uj30qh09m0sl.jpg width=800>

# 通过上面的例子可以看到，我们能够不断地将前面输出的文字重新输入到网络，不断循环递归，最后生成我们想要的长度的句子，是不是很简单呢？
# 
# 下面我们用 PyTorch 来具体实现

# 我们使用古诗来作为例子，读取这个数据，看看其长什么样子

# In[1]:


with open('./dataset/poetry.txt', 'r') as f:
    poetry_corpus = f.read()


# In[2]:


poetry_corpus[:100]


# In[3]:


# 看看字符数
len(poetry_corpus)


# 为了可视化比较方便，我们将一些其他字符替换成空格

# In[4]:


poetry_corpus = poetry_corpus.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')
poetry_corpus[:100]


# ### 文本数值表示
# 对于每个文字，电脑并不能有效地识别，所以必须做一个转换，将文字转换成数字，对所有非重复的字符，可以从 0 开始建立索引
# 
# 同时为了节省内存的开销，可以词频比较低的字去掉

# In[5]:


import numpy as np

class TextConverter(object):
    def __init__(self, text_path, max_vocab=5000):
        """建立一个字符索引转换器
        
        Args:
            text_path: 文本位置
            max_vocab: 最大的单词数量
        """
        
        with open(text_path, 'r') as f:
            text = f.read()
        text = text.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')
        # 去掉重复的字符
        vocab = set(text)

        # 如果单词总数超过最大数值，去掉频率最低的
        vocab_count = {}
        
        # 计算单词出现频率并排序
        for word in vocab:
            vocab_count[word] = 0
        for word in text:
            vocab_count[word] += 1
        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)
        
        # 如果超过最大值，截取频率最低的字符
        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab]
        vocab = [x[0] for x in vocab_count_list]
        self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)


# In[6]:


convert = TextConverter('./dataset/poetry.txt', max_vocab=10000)


# 我们可以可视化一下数字表示的字符

# In[7]:


# 原始文本字符
txt_char = poetry_corpus[:11]
print(txt_char)

# 转化成数字
print(convert.text_to_arr(txt_char))


# ### 构造时序样本数据
# 为了输入到循环神经网络中进行训练，我们需要构造时序样本的数据，因为前面我们知道，循环神经网络存在着长时依赖的问题，所以说我们不能将所有的文本作为一个序列一起输入到循环神经网络中，我们需要将整个文本分成很多很多个序列组成 batch 输入到网络中，只要我们定好每个序列的长度，那么序列个数也就被决定了。

# In[8]:


n_step = 20

# 总的序列个数
num_seq = int(len(poetry_corpus) / n_step)

# 去掉最后不足一个序列长度的部分
text = poetry_corpus[:num_seq*n_step]

print(num_seq)


# 接着我们将序列中所有的文字转换成数字表示，重新排列成 (num_seq x n_step) 的矩阵

# In[9]:


import torch


# In[10]:


arr = convert.text_to_arr(text)
arr = arr.reshape((num_seq, -1))
arr = torch.from_numpy(arr)

print(arr.shape)
print(arr[0, :])


# 据此，我们可以构建 PyTorch 中的数据读取来训练网络，这里我们将最后一个字符的输出 label 定为输入的第一个字符，也就是"床前明月光"的输出是"前明月光床"

# In[11]:


class TextDataset(object):
    def __init__(self, arr):
        self.arr = arr
        
    def __getitem__(self, item):
        x = self.arr[item, :]
        
        # 构造 label
        y = torch.zeros(x.shape)
        # 将输入的第一个字符作为最后一个输入的 label
        y[:-1], y[-1] = x[1:], x[0]
        return x, y
    
    def __len__(self):
        return self.arr.shape[0]


# In[12]:


train_set = TextDataset(arr)


# 我们可以取出其中一个数据集参看一下是否是我们描述的这样

# In[13]:


x, y = train_set[0]
print(convert.arr_to_text(x.numpy()))
print(convert.arr_to_text(y.numpy()))


# ### 建立模型
# 模型可以定义成非常简单的三层，第一层是词嵌入，第二层是 RNN 层，因为最后是一个分类问题，所以第三层是线性层，最后输出预测的字符。

# In[14]:


from torch import nn
from torch.autograd import Variable

use_gpu = True

class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, 
                 num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, dropout)
        self.project = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hs=None):
        batch = x.shape[0]
        if hs is None:
            hs = Variable(
                torch.zeros(self.num_layers, batch, self.hidden_size))
            if use_gpu:
                hs = hs.cuda()
        word_embed = self.word_to_vec(x)  # (batch, len, embed)
        word_embed = word_embed.permute(1, 0, 2)  # (len, batch, embed)
        out, h0 = self.rnn(word_embed, hs)  # (len, batch, hidden)
        le, mb, hd = out.shape
        out = out.view(le * mb, hd)
        out = self.project(out)
        out = out.view(le, mb, -1)
        out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden)
        return out.view(-1, out.shape[2]), h0


# ### 训练模型
# 在训练模型的时候，我们知道这是一个分类问题，所以可以使用交叉熵作为 loss 函数，在语言模型中，我们通常使用一个新的指标来评估结果，这个指标叫做困惑度(perplexity)，可以简单地看作对交叉熵取指数，这样其范围就是 $[1, +\infty]$，也是越小越好。
# 
# 另外，我们前面讲过 RNN 存在着梯度爆炸的问题，所以我们需要进行梯度裁剪，在 pytorch 中使用 `torch.nn.utils.clip_grad_norm` 就能够轻松实现

# In[15]:


from torch.utils.data import DataLoader

batch_size = 128
train_data = DataLoader(train_set, batch_size, True, num_workers=4)


# In[16]:


from mxtorch.trainer import ScheduledOptim

model = CharRNN(convert.vocab_size, 512, 512, 2, 0.5)
if use_gpu:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()

basic_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = ScheduledOptim(basic_optimizer)


# In[17]:


epochs = 20
for e in range(epochs):
    train_loss = 0
    for data in train_data:
        x, y = data
        y = y.long()
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        x, y = Variable(x), Variable(y)

        # Forward.
        score, _ = model(x)
        loss = criterion(score, y.view(-1))

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient.
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        train_loss += loss.data[0]
    print('epoch: {}, perplexity is: {:.3f}, lr:{:.1e}'.format(e+1, np.exp(train_loss / len(train_data)), optimizer.lr))


# 可以看到，训练完模型之后，我们能够到达 2.72 左右的困惑度，下面我们就可以开始生成文本了。
# 
# ### 生成文本
# 生成文本的过程非常简单，前面已将讲过了，给定了开始的字符，然后不断向后生成字符，将生成的字符作为新的输入再传入网络。
# 
# 这里需要注意的是，为了增加更多的随机性，我们会在预测的概率最高的前五个里面依据他们的概率来进行随机选择。

# In[18]:


def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


# In[19]:


begin = '天青色等烟雨'
text_len = 30

model = model.eval()
samples = [convert.word_to_int(c) for c in begin]
input_txt = torch.LongTensor(samples)[None]
if use_gpu:
    input_txt = input_txt.cuda()
input_txt = Variable(input_txt)
_, init_state = model(input_txt)
result = samples
model_input = input_txt[:, -1][:, None]
for i in range(text_len):
    out, init_state = model(model_input, init_state)
    pred = pick_top_n(out.data)
    model_input = Variable(torch.LongTensor(pred))[None]
    if use_gpu:
        model_input = model_input.cuda()
    result.append(pred[0])
text = convert.arr_to_text(result)
print('Generate text is: {}'.format(text))


# 最后可以看到，生成的文本已经想一段段的话了
