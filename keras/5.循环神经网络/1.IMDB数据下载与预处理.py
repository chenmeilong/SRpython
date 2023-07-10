#影评数据 50000条  预测是正面还是负面评价
import urllib.request
import os
import tarfile            #解压缩文件

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = "data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):                #下载数据集
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded:', result)
if not os.path.exists("data/aclImdb"):          #判断解压缩文件是否存在
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result = tfile.extractall('data/')

# 读取数据
from keras.preprocessing import sequence         #截长补短 数字列表
from keras.preprocessing.text import Tokenizer   #建立字典   语言字典
import re                                       #正则
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')           #删除HTML标签
    return re_tag.sub('', text)              #替换为空

def read_files(filetype):
    path = "data/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    return all_labels, all_texts

y_train, train_text = read_files("train")
y_test, test_text = read_files("test")

# 查看正面评价的影评
print (train_text[0])   #文字
print (y_train[0])       #标签
# 查看负面评价的影评
print (train_text[12501])
print (y_train[12501])

# # 先读取所有文章建立字典，限制字典的数量为nb_words=2000
token = Tokenizer(num_words=2000)           #建立2000词的字典
token.fit_on_texts(train_text)              #按照单词出现的次数排序  取前2000个单词

# Tokenizer属性
# fit_on_texts 读取多少文章
print(token.document_count)    #25000
# print(token.word_index)       #单词出现次数排名

# # 将每一篇文章的文字转换一连串的数字
# #只有在字典中的文字会转换为数字
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
print(train_text[0])
print(x_train_seq[0])

# # 让转换后的数字长度相同  100
x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)

# 如果文章转成数字大于0,pad_sequences处理后，会truncate前面的数字

print('before pad_sequences length=', len(x_train_seq[0]))
print(x_train_seq[0])
print('after pad_sequences length=', len(x_train[0]))
print(x_train[0])

