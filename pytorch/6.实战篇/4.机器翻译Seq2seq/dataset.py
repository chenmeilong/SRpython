import random
import re
import string
import unicodedata

import torch
from torch.utils.data import Dataset
import numpy as np

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 5           #训练词的最大长度


class Lang(object):                  #给每个不同单词打上标签
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):            #unicode转换成ascll 所有字母小写 ，去掉标点符号
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # 将每一行分成两对并进行标准化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    print(np.array(pairs).shape)                     #(135842, 2)

    # Reverse pairs, make Lang instances   是否 反向 英法---法英
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)      #实例化对象
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s",
                "you are", "you re ", "we are", "we re ", "they are",
                "they re ")


def filterPair(p):               #只保留 以某些开头的句子 两个句子都小于某个长度的  减少语句数量 提高训练速度
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print(np.array(pairs).shape)            #(1404, 2)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(random.choice(pairs))
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]    #返回的是一个词对应号的列表


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes)
    return result


def tensorFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


class TextDataset(Dataset):
    def __init__(self, dataload=prepareData, lang=['eng', 'fra']):
        self.input_lang, self.output_lang, self.pairs = dataload(
            lang[0], lang[1], reverse=True)     # reverse=True  语句调换位置  法输入  英输出
        self.input_lang_words = self.input_lang.n_words             #法语 单词个数
        self.output_lang_words = self.output_lang.n_words          #英语单词个数

    def __getitem__(self, index):                            #将语句转换成 tensor
        return tensorFromPair(self.input_lang, self.output_lang,
                              self.pairs[index])

    def __len__(self):
        return len(self.pairs)
