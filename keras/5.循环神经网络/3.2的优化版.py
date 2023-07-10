#单词嵌入层  + 全连接神经网络     和2结构相同   但增加词3800  长度为380     准确率0.84384
# # 数据准备
from keras.datasets import imdb
import os
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
np.random.seed(10)

import re     #去掉html 标签
re_tag = re.compile(r'<[^>]+>')
def rm_tags(text):
    return re_tag.sub('', text)

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
token = Tokenizer(num_words=3800)
token.fit_on_texts(train_text)       #按照单词出现的次数排序  取前3800单词  #转为词数字   字典
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)       #变成数字串
x_train = sequence.pad_sequences(x_train_seq, maxlen=380)     #大于380  就截取前面的文字  ，小于380就在前面补0
x_test = sequence.pad_sequences(x_test_seq, maxlen=380)

# # 建立模型
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding    #单词嵌入是使用密集的矢量表示来表示单词和文档的一类方法。
model = Sequential()
model.add(Embedding(output_dim=32,input_dim=3800, input_length=380))       #单词嵌入是使用密集的矢量表示来表示单词和文档的一类方法。
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()

# # 训练模型
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x_train, y_train, batch_size=100,epochs=10, verbose=2,validation_split=0.2)

#显示训练过程图像
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# # 评估模型的准确率
scores = model.evaluate(x_test, y_test, verbose=1)
print (scores[1])

# # 预测概率
probility = model.predict(x_test)
print (probility[:10])        #预测前个评价

# # 预测结果
predict = model.predict_classes(x_test)
print (predict[:10])
predict_classes = predict.reshape(-1)
print (predict_classes[:10])

# # 查看预测结果
SentimentDict = {1: '正面的', 0: '负面的'}
def display_test_Sentiment(i):
    print(test_text[i])
    print('标签label:', SentimentDict[y_test[i]],
          '预测结果:', SentimentDict[predict_classes[i]])
display_test_Sentiment(2)   #预测第三条


# 预测新的影评
input_text = '''
Oh dear, oh dear, oh dear: where should I start folks. I had low expectations already because I hated each and every single trailer so far, but boy did Disney make a blunder here. I'm sure the film will still make a billion dollars - hey: if Transformers 11 can do it, why not Belle? - but this film kills every subtle beautiful little thing that had made the original special, and it does so already in the very early stages. It's like the dinosaur stampede scene in Jackson's King Kong: only with even worse CGI (and, well, kitchen devices instead of dinos).
The worst sin, though, is that everything (and I mean really EVERYTHING) looks fake. What's the point of making a live-action version of a beloved cartoon if you make every prop look like a prop? I know it's a fairy tale for kids, but even Belle's village looks like it had only recently been put there by a subpar production designer trying to copy the images from the cartoon. There is not a hint of authenticity here. Unlike in Jungle Book, where we got great looking CGI, this really is the by-the-numbers version and corporate filmmaking at its worst. Of course it's not really a "bad" film; those 200 million blockbusters rarely are (this isn't 'The Room' after all), but it's so infuriatingly generic and dull - and it didn't have to be. In the hands of a great director the potential for this film would have been huge.
Oh and one more thing: bad CGI wolves (who actually look even worse than the ones in Twilight) is one thing, and the kids probably won't care. But making one of the two lead characters - Beast - look equally bad is simply unforgivably stupid. No wonder Emma Watson seems to phone it in: she apparently had to act against an guy with a green-screen in the place where his face should have been. 
'''

input_seq = token.texts_to_sequences([input_text])    #转换成数字列表
print (len(input_seq[0]))
pad_input_seq = sequence.pad_sequences(input_seq, maxlen=380)   #380  多删少补
print (len(pad_input_seq[0]))
predict_result = model.predict_classes(pad_input_seq)
print (predict_result[0][0])
print (SentimentDict[predict_result[0][0]])


def predict_review(input_text):            #将上面的打包成函数
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq, maxlen=380)
    predict_result = model.predict_classes(pad_input_seq)
    print(SentimentDict[predict_result[0][0]])

predict_review('''
It's hard to believe that the same talented director who made the influential cult action classic The Road Warrior had anything to do with this disaster.
Road Warrior was raw, gritty, violent and uncompromising, and this movie is the exact opposite. It's like Road Warrior for kids who need constant action in their movies.
This is the movie. The good guys get into a fight with the bad guys, outrun them, they break down in their vehicle and fix it. Rinse and repeat. The second half of the movie is the first half again just done faster.
The Road Warrior may have been a simple premise but it made you feel something, even with it's opening narration before any action was even shown. And the supporting characters were given just enough time for each of them to be likable or relatable.
In this movie there is absolutely nothing and no one to care about. We're supposed to care about the characters because... well we should. George Miller just wants us to, and in one of the most cringe worthy moments Charlize Theron's character breaks down while dramatic music plays to try desperately to make us care.
Tom Hardy is pathetic as Max. One of the dullest leading men I've seen in a long time. There's not one single moment throughout the entire movie where he comes anywhere near reaching the same level of charisma Mel Gibson did in the role. Gibson made more of an impression just eating a tin of dog food. I'm still confused as to what accent Hardy was even trying to do.
I was amazed that Max has now become a cartoon character as well. Gibson's Max was a semi-realistic tough guy who hurt, bled, and nearly died several times. Now he survives car crashes and tornadoes with ease?
In the previous movies, fuel and guns and bullets were rare. Not anymore. It doesn't even seem Post-Apocalyptic. There's no sense of desperation anymore and everything is too glossy looking. And the main villain's super model looking wives with their perfect skin are about as convincing as apocalyptic survivors as Hardy's Australian accent is. They're so boring and one-dimensional, George Miller could have combined them all into one character and you wouldn't miss anyone.
Some of the green screen is very obvious and fake looking, and the CGI sandstorm is laughably bad. It wouldn't look out of place in a Pixar movie.
There's no tension, no real struggle, or any real dirt and grit that Road Warrior had. Everything George Miller got right with that masterpiece he gets completely wrong here. 
''')
