#保存图片  显示训练图片
import PIL.Image
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片

def showarray(a):
    a = np.uint8(np.clip(a, 0, 255))
    im =PIL.Image.fromarray(a)
    im.save("out.jpeg")

    plt.imshow(a)
    plt.show()                  #        图片

def showtensor(a):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std * inp + mean
    inp *= 255
    showarray(inp)
    clear_output(wait=True)
