#分析url timestamp 和_signature 可忽略 这是做验证的
#找到 ajax 的 URL 规律  再找到图片的URL  爬取保存到本地目录
import requests,re,os
from hashlib import md5   #加密模块  在本代码中起到随机生成图片名字的作用
from selenium import webdriver

def get_cookies(url):
    str=''
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    browser = webdriver.Chrome(options=options)
    browser.get(url)
    for i in browser.get_cookies():
        try:
            name=i.get('name')
            value=i.get('value')
            str=str+name+'='+value+';'
        except ValueError as e:
            print(e)
    return str

def get_page(offset):
    params = {
        'aid': '24',
        'app_name': 'web_search',
        'offset': offset,
        'format': 'json',
        'keyword': '街拍',
        'autoload': 'true',
        'count': '20',
        'en_qc': '1',
        'cur_tab': '1',
        'from': 'search_tab',
        'pd': 'synthesis',
    }
    url='https://www.toutiao.com/api/search/content/'
    try:
        r=requests.get(url,params=params,headers=headers)
        if r.status_code==200:
            return r.json()
        else:
            print('requests get_page error!')
    except requests.ConnectionError:
        return None

def get_images(json):
    data=json.get('data')
    if data:
        for i in data:
            if i.get('title'):
                title=re.sub('[\t]','',i.get('title'))   #re.sub(字符串里需要替换的东西,替换成成这个,需要替换的字符串)
                url=i.get('article_url')
                if url:
                    r=requests.get(url,headers=headers)
                    if r.status_code==200:
                        images_pattern = re.compile('JSON.parse\("(.*?)"\),\n', re.S)   #   re.S 对换行有效
                        result = re.search(images_pattern, r.text)   #一大堆 似乎为二进制文件
                        if result:
                            b_url='http://p3.pstatp.com/origin/pgc-image/'
                            up=re.compile('url(.*?)"width',re.S)   #贪婪模式
                            results=re.findall(up,result.group(1))
                            if results:
                                for result in results:
                                    yield {
                                        'title':title,
                                        'image':b_url+re.search('F([^F]*)\\\\",',result).group(1)   #这里貌似拼成了个图片URL
                                    }
                        else:
                            images = i.get('image_list')
                            for image in images:
                                origin_image = re.sub("list.*?pgc-image", "large/pgc-image",image.get('url'))  # 将小图改成原图路径
                                yield {
                                    'image': origin_image,
                                    'title': title
                                }

def save_image(item):
    img_path = 'img' + os.path.sep + item.get('title')
    if not os.path.exists(img_path):
        os.makedirs(img_path) # 生成目录文件夹
    try:
        resp = requests.get(item.get('image'))  #item.get('image')是图片的路径
        if requests.codes.ok == resp.status_code:
            file_path = img_path + os.path.sep + '{file_name}.{file_suffix}'.format(
                file_name=md5(resp.content).hexdigest(),  #把任意长度的输入，通过某种hash算法，变换成固定长度的输出，该算法就是哈希函数
                file_suffix='jpg')  # 单一文件的路径  os.path.sep路径分隔符
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    f.write(resp.content)
                print('Downloaded image path is %s' % file_path)
            else:
                print('Already Downloaded', file_path)
        else:
            print('这个URL无效')
    except Exception as e:
        print(e,'none123')

def main(offset):     #0,20,40
    a = get_page(offset)
    for i in get_images(a):
        save_image(i)

cookies = get_cookies('https://www.toutiao.com')
headers = {
    'cookie': cookies,
    'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest',
    'referer': 'https://www.toutiao.com/search/?keyword=%E8%A1%97%E6%8B%8D',   #街拍URL
}

if __name__=='__main__':
    for i in [x*20 for x in range(3)]:    #0 20 40 每页20个内容
        main(i)
