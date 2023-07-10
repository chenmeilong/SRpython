# 登录成功

import requests
from lxml import etree


class Login(object):
    def __init__(self):
        self.headers = {
            'Referer': 'https://github.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
            'Host': 'github.com'
        }
        self.login_url = 'https://github.com/login'
        self.post_url = 'https://github.com/session'
        self.logined_url = 'https://github.com/settings/profile'
        self.session = requests.Session()

    def token(self):
        response = self.session.get(self.login_url, headers=self.headers)
        selector = etree.HTML(response.text)
        token = selector.xpath('//*[@id="login"]/form/input[1]/@value')  # 解析出 authenticity_token的值
        return token

    def dynamics(self, html):
        selector = etree.HTML(html)  # 提取所有动态信息
        dynamics = selector.xpath('//div[contains(@class, "news")]//div[contains(@class, "alert")]')
        for item in dynamics:
            dynamic = ' '.join(item.xpath('.//div[@class="title"]//text()')).strip()
            print(dynamic)

    def profile(self, html):
        selector = etree.HTML(html)
        name = selector.xpath('//*[@id="user_profile_name"]')
        email = selector.xpath('//*[@id="user_profile_email"]/text()')
        print(name, email)

    def login(self, email, password):
        post_data = {
            'commit': 'Sign in',
            'utf8': '✓',
            'authenticity_token': self.token()[0],
            'login': email,
            'password': password
        }
        response = self.session.post(self.post_url, data=post_data, headers=self.headers)  # post 提交
        print("提交后返回的状态码:", response.status_code)
        if response.status_code == 200:
            self.dynamics(response.text)
            print('打印动态信息')

        response = self.session.get(self.logined_url, headers=self.headers)
        if response.status_code == 200:
            self.profile(response.text)
            print("打印基础信息完成")


if __name__ == "__main__":
    login = Login()
    login.login(email='chenmeilong', password='1314520WWw')
