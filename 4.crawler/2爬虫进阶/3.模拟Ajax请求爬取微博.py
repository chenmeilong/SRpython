# 微博原来自动ajax加载page 改成了since_id  每页data的since_id即是下一页的since_id

import requests
from urllib.parse import urlencode
from pyquery import PyQuery as pq

base_url = 'https://m.weibo.cn/api/container/getIndex?'
headers = {
    'Host': 'm.weibo.cn',
    'Referer': 'https://m.weibo.cn/u/2830678474',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}
max_page = 10
since_id=0

def get_page():
    if since_id==0:
        params = {
            'type': 'uid',
            'value': '2830678474',
            'containerid': '1076032830678474',
        }
    else:
        params = {
            'type': 'uid',
            'value': '2830678474',
            'containerid': '1076032830678474',
            'since_id':since_id
        }
    url = base_url + urlencode(params)    #字典转成URl
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            #print( response.json())
            return response.json()
    except requests.ConnectionError as e:
        print('Error', e.args)


def parse_page(json):   #因为有 yield返回值  这个函数就是一个生成器
    global since_id
    if json:
        items = json.get('data').get('cards')
        since_id=json.get('data').get('cardlistInfo').get('since_id')  #拿到since_id
        for index, item in enumerate(items):
            if index == 1:
                continue
            else:
                item = item.get('mblog', {})
                weibo = {}
                weibo['id'] = item.get('id')
                weibo['text'] = pq(item.get('text')).text()   #使用pyquery将正文中的HTML标签去掉
                weibo['attitudes'] = item.get('attitudes_count')
                weibo['comments'] = item.get('comments_count')
                weibo['reposts'] = item.get('reposts_count')
                yield weibo

if __name__ == '__main__':
    for page in range(1, max_page + 1):
        json = get_page()
        results = parse_page(json)
        for result in results:
            print(result)
