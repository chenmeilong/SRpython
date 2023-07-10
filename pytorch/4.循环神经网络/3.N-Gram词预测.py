# # N-Gram 模型

# 对于一句话，单词的排列顺序是非常重要的，所以我们能否由前面的几个词来预测后面的几个单词呢，
# 比如 'I lived in France for 10 years, I can speak _' 这句话中，我们能够预测出最后一个词是 French。

# 我们可以再简化一下这个模型，比如对于一个词，它并不需要前面所有的词作为条件概率，也就是说一个词可以只与其前面的几个词有关，这就是马尔科夫假设。

# 对于这里的条件概率，传统的方法是统计语料中每个词出现的频率，根据贝叶斯定理来估计这个条件概率，这里我们就可以用词嵌入对其进行代替，
# 然后使用 RNN 进行条件概率的计算，然后最大化这个条件概率不仅修改词嵌入，同时能够使得模型可以依据计算的条件概率对其中的一个单词进行预测。
#

CONTEXT_SIZE = 2  # 依据的单词数    表示我们希望由前面几个单词来预测这个单词，这里使用两个单词
EMBEDDING_DIM = 10  # 词向量的维度
# 我们使用莎士比亚的诗
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# 接着我们建立训练集，便利整个语料库，将单词三个分组，前面两个作为输入，最后一个作为预测的结果。
trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]       #len(test_sentence) =115

print (len(trigram))      # 总的数据量
print (trigram[0])    # 取出第一个数据看看

# 建立每个词与数字的编码，据此构建词嵌入
vocb = set(test_sentence)  # 使用 set 将重复的元素去掉   字典 无号
word_to_idx = {word: i for i, word in enumerate(vocb)}   #enumerate函数用于将一个可遍历的数据对象组合为一个索引序列。同时列出数据和数据下标。
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}    #字典位置互换
print (word_to_idx)
print (idx_to_word)

# 从上面可以看到每个词都对应一个数字，且这里的单词都各不相同
# 接着我们定义模型，模型的输入就是前面的两个词，输出就是预测单词的概率

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


# 定义模型
class n_gram(nn.Module):
    def __init__(self, vocab_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM):
        super(n_gram, self).__init__()

        self.embed = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )      #torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。

    def forward(self, x):
        voc_embed = self.embed(x)  # 得到词嵌入
        voc_embed = voc_embed.view(1, -1)  # 将两个词向量拼在一起
        out = self.classify(voc_embed)
        return out

# 最后我们输出就是条件概率，相当于是一个分类问题，我们可以使用交叉熵来方便地衡量误差
net = n_gram(len(word_to_idx))    #len(word_to_idx)   ocab_size    实例化网络
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-5)

for e in range(100):
    train_loss = 0
    for word, label in trigram:  # 使用前 100 个作为训练集
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))  # 将两个词作为输入
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # 前向传播
        out = net(word)
        loss = criterion(out, label)
        train_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 20 == 0:
        print('epoch: {}, Loss: {:.6f}'.format(e + 1, train_loss / len(trigram)))

# 最后我们可以测试一下结果
net = net.eval()   #开始测试

# 测试一下结果
word, label = trigram[19]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].data[0]
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))


word, label = trigram[75]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].data[0]
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))

# 可以看到网络在训练集上基本能够预测准确，不过这里样本太少，特别容易过拟合。
