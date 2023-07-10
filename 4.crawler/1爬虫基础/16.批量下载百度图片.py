import re
import requests
from urllib import error
from bs4 import BeautifulSoup
import os


from selenium import webdriver


numPicture = 200
file = ''

def Find(url):
    print('正在检测图片总数，请稍等.....')
    s = 0
    try:
        driver = webdriver.Firefox()
        driver.set_window_size(1000, 30000)
        driver.get(url)
        result = driver.find_element_by_xpath("//*").get_attribute("outerHTML")
    except BaseException:
        print("页面加载不出来")
    else:
        pic_url = re.findall('"objURL":"(.*?)",', result, re.S)  # 先利用正则表达式找到图片url
        s += len(pic_url)
    return s


def recommend(url):
    Re = []
    try:
        html = requests.get(url)
    except error.HTTPError as e:
        return
    else:
        html.encoding = 'utf-8'
        bsObj = BeautifulSoup(html.text, 'html.parser')
        div = bsObj.find('div', id='topRS')
        if div is not None:
            listA = div.findAll('a')
            for i in listA:
                if i is not None:
                    Re.append(i.get_text())
        return Re


def dowmloadPicture(html, keyword):
    num=0
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)  # 先利用正则表达式找到图片url
    print('找到关键词:' + keyword + '的图片，即将开始下载图片...')
    for each in pic_url:
        print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
        try:
            if each is not None:
                pic = requests.get(each, timeout=7)
            else:
                continue
        except BaseException:
            print('错误，当前图片无法下载')
            continue
        else:
            string = file + r'\\' + keyword + '_' + str(num) + '.jpg'
            fp = open(string, 'wb')
            fp.write(pic.content)
            fp.close()
            num += 1
        if num >= numPicture:
            return


if __name__ == '__main__':  # 主函数入口
    word="折叠空间"

    url = 'https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gb18030&word=%D5%DB%B5%FE%BF%D5%BC%E4&fr=ala&ala=1&alatpl=adress&pos=0&hs=2&xthttps=111111'
    tot = Find(url)
    Recommend = recommend(url)  # 记录相关推荐
    print('经过检测%s类图片共有%d张' % (word, tot))
    file = word + '文件'
    y = os.path.exists(file)
    if y == 1:
        print('该文件已存在，请重新输入')
        file = word + '文件夹2'
        os.mkdir(file)
    else:
        os.mkdir(file)
    try:

        driver = webdriver.Firefox()
        driver.set_window_size(1000, 30000)
        driver.get(url)
        result = driver.find_element_by_xpath("//*").get_attribute("outerHTML")
        #print(result.text,"!!!!!", word)
    except error.HTTPError as e:
        print('网络错误，请调整网络后重试')
    else:
        dowmloadPicture(result, word)

    print('当前搜索结束')