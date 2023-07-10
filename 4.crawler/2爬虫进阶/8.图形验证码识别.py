import tesserocr
from PIL import Image
image = Image.open('data/code.jpg')
result = tesserocr.image_to_text(image)
print(result)                             #验证码有很多线条所以有错误
print(tesserocr.file_to_text('data/code.jpg'))   #这种方法准确率不高  不推荐使用
#二值化
image = image.convert('L')  #转为灰度图像
threshold = 127
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)

image = image.point(table, '1')
image.show()
result = tesserocr.image_to_text(image)
print(result)


