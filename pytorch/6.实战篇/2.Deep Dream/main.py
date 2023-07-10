#主要执行这个代码      gpu 运行 图片不能过大  cpu运行偏慢
import torch
from torchvision import transforms
from PIL import Image
from resnet import resnet50
from deepdream import dream

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])                                                           #数据增强

input_img = Image.open('./cat.jpg')
input_tensor = img_transform(input_img).unsqueeze(0)
input_np = input_tensor.numpy()          #(1, 3, 1059, 1887)

# load model
model = resnet50(pretrained=True)           #加载模型和参数   50层残差网络
if torch.cuda.is_available():
    model = model.cuda()
for param in model.parameters():
    param.requires_grad = False              #冻结模型层

dream(model, input_np)

# ## Control the dream
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import os
from resnet import resnet50
from deepdream import dream
from PIL import Image
from util import showtensor

img_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])                 #数据增强
inputs_control = Image.open('./guide_image/flower.jpg')
inputs_control = img_transform(inputs_control).unsqueeze(0)
inputs_control_np = inputs_control.numpy()

showtensor(inputs_control_np)

model = resnet50(pretrained=True)
if torch.cuda.is_available():
    model = model.cuda()
for param in model.parameters():
    param.requires_grad = False

if torch.cuda.is_available():
    x_variable = Variable(inputs_control.cuda())
else:
    x_variable = Variable(inputs_control)

control_features = model.forward(x_variable, end_layer=3)

def objective_guide(dst, guide_features):
    x = dst.data[0].cpu().numpy().copy()
    y = guide_features.data[0].cpu().numpy()
    ch, w, h = x.shape
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    result = y[:,A.argmax(1)] # select ones that match best
    result = torch.Tensor(np.array([result.reshape(ch, w, h)], dtype=np.float)).cuda()
    return result

dream(model, input_np, control=control_features, distance=objective_guide)
