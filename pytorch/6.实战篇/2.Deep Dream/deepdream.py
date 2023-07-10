#训练模块  这里可以调节训练参数
import numpy as np
import torch
from util import showtensor
import scipy.ndimage as nd
from torch.autograd import Variable

def objective_L2(dst, guide_features):
    return dst.data


def make_step(img, model, control=None, distance=objective_L2):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    learning_rate = 2e-2
    max_jitter = 32            #最大抖动
    num_iterations = 20         #迭代次数

    end_layer = 3
    guide_features = control

    for i in range(num_iterations):
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)
        # apply jitter shift
        model.zero_grad()
        img_tensor = torch.Tensor(img)
        if torch.cuda.is_available():
            img_variable = Variable(img_tensor.cuda(), requires_grad=True)
        else:
            img_variable = Variable(img_tensor, requires_grad=True)

        act_value = model.forward(img_variable, end_layer)
        diff_out = distance(act_value, guide_features)
        act_value.backward(diff_out)
        ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
        learning_rate_use = learning_rate / ratio
        img_variable.data.add_(img_variable.grad.data * learning_rate_use)
        img = img_variable.data.cpu().numpy()  # b, c, h, w
        img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
        img[0, :, :, :] = np.clip(img[0, :, :, :], -mean / std,           #图片逆抖动
                                  (1 - mean) / std)
    return img


def dream(model,
          base_img,
          octave_n=6,           #可调 多尺度图片数目
          octave_scale=1.4,
          control=None,
          distance=objective_L2):
    show_every = 2           #显示图片
    octaves = [base_img]           #  #[(1, 3, 1059, 1887)]
    for i in range(octave_n - 1):    #上采样与下采样 缩小与放大 使用多尺度图片
        octaves.append(
            nd.zoom(
                octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale),
                order=1))

    detail = np.zeros_like(octaves[-1])           #函数主要是想实现构造一个矩阵W_update，其维度与矩阵W一致，并为其初始化为全0
    for octave, octave_base in enumerate(octaves[::-1]):
        print (octave)
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(
                detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        input_oct = octave_base + detail        #(1, 3, 197, 351)
        out = make_step(input_oct, model, control, distance=distance)
        if octave == 0 or (octave + 1) % show_every == 0:
            showtensor(out)
        detail = out - octave_base
