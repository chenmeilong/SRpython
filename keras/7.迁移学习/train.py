#迁移学习   使用inception_v3模型的权重   1. 后面增加全连接网络 5 类全连接  2.在1的基础上冻结前n层参数迁移学习
# 此代码涉及图片数据增强。      模型路径C:\Users\Wayne\.keras
import os
import sys
import glob    #用它可以查找符合特定规则的文件路径名
import argparse
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator          #数据增强
from keras.optimizers import SGD

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


# 数据准备
IM_WIDTH, IM_HEIGHT = 299, 299 #InceptionV3指定的图片尺寸
FC_SIZE = 1024                # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 172  # 冻结层的数量


train_dir = 'data\\train'  # 训练集数据
val_dir = 'data\\test' # 验证集数据
nb_classes= 5
nb_epoch = 3
batch_size = 20

nb_train_samples = get_nb_files(train_dir)      # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)       #验证集样本个数
nb_epoch = int(nb_epoch)                # epoch数量
batch_size = int(batch_size)

#　图片生成器      图像数据增强
train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocess_input,           #图片大小
  rotation_range=30,                                #整数。随机旋转的度数范围。
  width_shift_range=0.2,                             #分别是水平位置评议和上下位置平移
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True )
test_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True )

# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=batch_size,class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
val_dir,
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=batch_size,class_mode='categorical')


# 添加新层
def add_new_last_layer(base_model, nb_classes):                 ###############重点
  """
  添加最后的层
  输入
  base_model和分类数量
  输出
  新的keras的model
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)         #对于空域数据的全局平均池化。
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  # print(model.summary())
  return model

# 冻上base_model所有层，这样就可以正确获得bottleneck特征############################################################
def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False                           #迁移学习的重点重点
  # print(model.summary())
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   # 定义训练方式

# 定义网络框架
base_model = InceptionV3(weights='imagenet', include_top=False) # 预先要下载no_top模型
model = add_new_last_layer(base_model, nb_classes)              # 从基本no_top模型上添加新层
setup_to_transfer_learn(model, base_model)                      # 冻结base_model所有层

# 模式一训练
history_tl = model.fit_generator(
train_generator,
nb_epoch=nb_epoch,
samples_per_epoch=nb_train_samples,                  #训练数据个数
validation_data=validation_generator,
nb_val_samples=nb_val_samples,
class_weight='auto')
#利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练


# 冻上NB_IV3_LAYERS之前的层    冻结前面n层参数
def setup_to_finetune(model):
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  # print(model.summary())     Total params: 7,690,149      Trainable params: 2,103,301
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# 设置网络结构
setup_to_finetune(model)

# 模式二训练
history_ft = model.fit_generator(
train_generator,
samples_per_epoch=nb_train_samples,
nb_epoch=nb_epoch,
validation_data=validation_generator,
nb_val_samples=nb_val_samples,
class_weight='auto')

# 模型保存
model.save("data/inception_v3_transfer.h5")  # 保存模型和权重

# 画图
def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')
  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()

# 训练的acc_loss图
plot_training(history_ft)

