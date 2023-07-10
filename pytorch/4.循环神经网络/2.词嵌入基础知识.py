# # 词嵌入

# ## PyTorch 实现
# 词嵌入在 pytorch 中非常简单，只需要调用 `torch.nn.Embedding(m, n)` 就可以了，m 表示单词的总数目，n 表示词嵌入的维度，其实词嵌入就相当于是一个大矩阵，矩阵的每一行表示一个单词

import torch
from torch import nn
from torch.autograd import Variable

# 定义词嵌入
embeds = nn.Embedding(2, 5) # 2 个单词，维度 5\
print (embeds.weight)  # 得到词嵌入矩阵

# 我们通过 `weight` 得到了整个词嵌入的矩阵，注意，这个矩阵是一个可以改变的 parameter，在网络的训练中会不断更新，
# 同时词嵌入的数值可以直接进行修改，比如我们可以读入一个预训练好的词嵌入等等

embeds.weight.data = torch.ones(2, 5)   # 直接手动修改词嵌入的值
print (embeds.weight)

# 访问第 50 个词的词向量
embeds = nn.Embedding(100, 10)
single_word_embed = embeds(Variable(torch.LongTensor([50])))
print (single_word_embed)

# 可以看到如果我们要访问其中一个单词的词向量，我们可以直接调用定义好的词嵌入，但是输入必须传入一个 Variable，且类型是 LongTensor
# 虽然我们知道了如何定义词向量的相似性，但是我们仍然不知道如何得到词嵌入，因为如果一个词嵌入式 100 维，这显然不可能人为去赋值，所以为了得到词向量，需要介绍 skip-gram 模型。

# ## Skip-Gram 模型
# Skip Gram 模型是 [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf) 这篇论文的网络架构，下面我们来讲一讲这个模型。
# ## 模型结构
# skip-gram 模型非常简单，我们在一段文本中训练一个简单的网络，这个网络的任务是通过一个词周围的词来预测这个词，然而我们实际上要做的就是训练我们的词嵌入。
#
# 比如我们给定一句话中的一个词，看看它周围的词，然后随机挑选一个，我们希望网络能够输出一个概率值，这个概率值能够告诉我们到底这个词离我们选择的词的远近程度，比如这么一句话 'A dog is playing with a ball'，如果我们选的词是 'ball'，那么 'playing' 就要比 'dog' 离我们选择的词更近。
#
# 对于一段话，我们可以按照顺序选择不同的词，然后构建训练样本和 label，比如
#
# ![](https://ws2.sinaimg.cn/large/006tNc79gy1fmwlpfp3loj30hh0ah75l.jpg)

# 对于这个例子，我们依次取一个词以及其周围的词构成一个训练样本，比如第一次选择的词是 'the'，那么我们取其前后两个词作为训练样本，
# 这个也可以被称为一个滑动窗口，对于第一个词，其左边没有单词，所以训练集就是三个词，然后我们在这三个词中选择 'the' 作为输入，
# 另外两个词都是他的输出，就构成了两个训练样本，又比如选择 'fox' 这个词，那么加上其左边两个词，右边两个词，一共是 5 个词，
# 然后选择 'fox' 作为输入，那么输出就是其周围的四个词，一共可以构成 4 个训练样本，通过这个办法，我们就能够训练出需要的词嵌入。
#