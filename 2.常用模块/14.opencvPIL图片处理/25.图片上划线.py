
from PIL import Image, ImageDraw

myimg = Image.open('data/cutcard/7.png')  # 打开图片

print(myimg .size) ## 打印出尺寸信息
region = myimg.crop((0, 270, 793, 330))

region =region .resize((690, 46))
print(region .size) ## 打印出尺寸信息


x = 30  # 第一个点列坐标
y1 = 0  #第一个点行坐标
y2 = 46 #第二个点行坐标

draw = ImageDraw.Draw(region) #导入模块
draw.line([x,y1,x,y2], fill = (255, 0, 0), width = 1)
draw.line([x*2,y1,x*2,y2], fill = (255, 0, 0), width = 1)
draw.line([x*3,y1,x*3,y2], fill = (255, 0, 0), width = 1)
draw.line([x*4,y1,x*4,y2], fill = (255, 0, 0), width = 1)
draw.line([x*5,y1,x*5,y2], fill = (255, 0, 0), width = 1)
draw.line([x*6,y1,x*6,y2], fill = (255, 0, 0), width = 1)
draw.line([x*7,y1,x*7,y2], fill = (255, 0, 0), width = 1)
draw.line([x*8,y1,x*8,y2], fill = (255, 0, 0), width = 1)
draw.line([x*9,y1,x*9,y2], fill = (255, 0, 0), width = 1)
draw.line([x*10,y1,x*10,y2], fill = (255, 0, 0), width = 1)
draw.line([x*11,y1,x*11,y2], fill = (255, 0, 0), width = 1)
draw.line([x*12,y1,x*12,y2], fill = (255, 0, 0), width = 1)
draw.line([x*13,y1,x*13,y2], fill = (255, 0, 0), width = 1)
draw.line([x*14,y1,x*14,y2], fill = (255, 0, 0), width = 1)
draw.line([x*15,y1,x*15,y2], fill = (255, 0, 0), width = 1)
draw.line([x*16,y1,x*16,y2], fill = (255, 0, 0), width = 1)
draw.line([x*17,y1,x*17,y2], fill = (255, 0, 0), width = 1)
draw.line([x*18,y1,x*18,y2], fill = (255, 0, 0), width = 1)
draw.line([x*19,y1,x*19,y2], fill = (255, 0, 0), width = 1)
draw.line([x*20,y1,x*20,y2], fill = (255, 0, 0), width = 1)
draw.line([x*21,y1,x*21,y2], fill = (255, 0, 0), width = 1)
draw.line([x*22,y1,x*22,y2], fill = (255, 0, 0), width = 1)





region .show()




