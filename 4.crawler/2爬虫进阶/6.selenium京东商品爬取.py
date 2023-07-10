#京东爬取商品    换页点击 有时好使 有时不好使 应该是网络问题
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from pyquery import PyQuery as pq
from urllib.parse import quote
import time
KEYWORD = 'ipad'
MAX_PAGE = 1
SERVICE_ARGS = ['--load-images=false', '--disk-cache=true']

# browser = webdriver.Chrome()


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
browser = webdriver.Chrome(chrome_options=chrome_options)      #可以不打开 浏览器窗口


wait = WebDriverWait(browser,10)

def index_page(page):
    """ :param page: 页码"""
    print('正在爬取第', page, '页')
    try:
        url = 'https://search.jd.com/Search?keyword=' + quote(KEYWORD)   #quote 中文转URL编码
        browser.get(url)
        if page > 1:
            input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.p-wrap .p-skip .input-txt')))
            #判断是否至少有1个元素存在于dom树中，如果定位到就返回列表   #mainsrp-pager div.form > input
            submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.p-wrap .p-skip .btn')))
            #根据css选择器查找按钮是否被加载        #mainsrp-pager div.form > span.btn.J_Submit
            input.clear()
            input.send_keys(str(page))
            submit.click()
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.m-list .ml-wrap .page')))   #等待节点加载出来
        wait.until(EC.text_to_be_present_in_element_value((By.CSS_SELECTOR, '.p-wrap .p-skip .input-txt'), str(page)))  #  _value 输入框包含某文字
        #'''判断指定的元素中是否包含了预期的字符串，返回布尔值'''
        get_products()
    except TimeoutException:
        print("超时，没有爬取成功")


def get_products():
    """提取商品数据"""
    html = browser.page_source
    doc = pq(html)           #使用pyquery解析html
    items = doc('#J_goodsList .gl-warp .gl-item').items()
    for item in items:           #单个商品的HTML
        product = {
            'image': item.find('.p-img').find('img').attr('src'),        #查找所有子孙节点
            'price': item.find('.p-price').find('i').text(),             #价格
            'evaluate': item.find('.p-commit').find('strong') .find('a').text(),                  #评价数量
            'title': item.find('.p-name-type-2').find('a').text(),
            'shop': item.find('.p-shop').find('a').text(),
        }
        print(product)

if __name__ == '__main__':
    #遍历每一页
    for i in range(1, MAX_PAGE + 1):
        index_page(i)
    # browser.close()
