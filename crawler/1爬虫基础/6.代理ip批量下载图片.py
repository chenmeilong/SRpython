import urllib.request                 #使用时需要更换代理ip        OOXX文件夹
import urllib.request                 #使用时需要更换代理ip        OOXX文件夹
import os
import random


def url_open(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36')

    proxies = ['118.190.145.138:9001', '117.191.11.77:8080', '101.4.136.34:8080']       #
    proxy = random.choice(proxies)

    proxy_support = urllib.request.ProxyHandler({'http':proxy})
    opener = urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)

    response = urllib.request.urlopen(url)
    html = response.read()

    return html


def get_page(url):
    html = url_open(url).decode('utf-8')

    a = html.find('current-comment-page') + 23
    b = html.find(']', a)


    print (html[a:b])

    return html[a:b]


def find_imgs(url):
    html = url_open(url).decode('utf-8')
    img_addrs = []

    a = html.find('img src=')

    while a != -1:
        b = html.find('.jpg', a, a+255)
        if b != -1:
            img_addrs.append(html[a+11:b+4])
        else:
            b = a + 9

        a = html.find('img src=', b)

    return img_addrs


def save_imgs(folder, img_addrs):
    print (img_addrs)
    for each in img_addrs:
        filename = each.split('/')[-1]
        response = urllib.request.urlopen("http://"+each)
        img = response.read()
        with open(filename, 'wb') as f:
            f.write(img)


def download_mm(folder='OOXX', pages=10):
    os.mkdir(folder)
    os.chdir(folder)

    url = "http://jandan.net/ooxx/"
    page_num = int(get_page(url))

    for i in range(pages):
        page_num -= i
        page_url = url + 'page-' + str(page_num) + '#comments'
        img_addrs = find_imgs(page_url)
        save_imgs(folder, img_addrs)

if __name__ == '__main__':
    download_mm()
