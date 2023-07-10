#执行这个文件
from torch.autograd import Variable
from torchvision import transforms
from run_code import run_style_transfer
from load_img import load_img, show_img
from torch.autograd import Variable
import matplotlib.pyplot as plt # plt 用于显示图片

style_img = load_img('./picture/style.png')
style_img = Variable(style_img).cuda()
content_img = load_img('./picture/content.png')
content_img = Variable(content_img).cuda()
input_img = content_img.clone()
out = run_style_transfer(content_img, style_img, input_img, num_epoches=300)

plt.imshow(transforms.ToPILImage()(out.cpu().squeeze(0)))
plt.show()  # 图片
save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))          #转化为PILImage并显示   squeeze从数组的形状中删除单维度条目，即把shape中为1的维度去掉
save_pic.save('./picture/saved_picture1.png')
