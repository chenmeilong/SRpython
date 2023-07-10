# 定义层
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

# 狂阶图片指定尺寸
target_size = (229, 229) #fixed size for InceptionV3 architecture

# 预测函数
# 输入：model，图片，目标尺寸
# 输出：预测predict
def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]


# 载入模型
model = load_model("data\\inception_v3_transfer.h5")

img = Image.open("cat.jpg")
preds = predict(model, img, target_size)
print (preds)


# 本地图片
img = Image.open("elephant.jpg")
preds = predict(model, img, target_size)
print (preds)


img = Image.open("flower.jpg")
preds = predict(model, img, target_size)
print (preds)

img = Image.open("horse.jpg")
preds = predict(model, img, target_size)
print (preds)

