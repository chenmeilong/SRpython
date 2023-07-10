# 网站更换验证方式了， 以后需要更加实际情况更改代码
import time
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

EMAIL = 'cqc@cuiqingcai.com'
PASSWORD = '123456'
BORDER = 6
INIT_LEFT = 60


class CrackGeetest():
    def __init__(self):  # 打开网页
        self.url = 'https://account.geetest.com/login'
        self.browser = webdriver.Chrome()
        self.wait = WebDriverWait(self.browser, 20)
        self.email = EMAIL
        self.password = PASSWORD

    def __del__(self):
        self.browser.close()

    def get_geetest_button(self):
        """显示等待的方法 获取初始验证按钮"""
        button = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'geetest_radar_tip')))
        return button

    def get_position(self):
        """
        获取验证码位置       原始图片
        :return: 验证码位置元组   左上 右下角坐标
        """
        img = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'geetest_canvas_img')))
        time.sleep(2)
        location = img.location
        size = img.size
        top, bottom, left, right = location['y'], location['y'] + size['height'], location['x'], location['x'] + size[
            'width']
        return (top, bottom, left, right)

    def get_screenshot(self):
        """
        获取网页截图      #整个网页截图
        :return: 截图对象
        """
        screenshot = self.browser.get_screenshot_as_png()
        screenshot = Image.open(BytesIO(screenshot))
        return screenshot

    def get_slider(self):
        """
        获取滑块
        :return: 滑块对象
        """
        slider = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'geetest_slider_button')))
        return slider

    def get_geetest_image(self, name='captcha.png'):
        """
        获取验证码图片
        :return: 图片对象
        """
        top, bottom, left, right = self.get_position()
        print('验证码位置', top, bottom, left, right)
        screenshot = self.get_screenshot()
        captcha = screenshot.crop((left, top, right, bottom))  # 根据前面的验证码位置和截图  剪裁图片
        captcha.save(name)
        return captcha

    def open(self):
        """
        打开网页输入用户名密码
        :return: None
        """
        self.browser.get(self.url)

        # web-content > div > div.tyc-home-top.bgtyc > div.mt-74 > div.tyc-header.-home > div > div.right > div > div:nth-child(5) > a

        email = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                                '#base > div.content-outter > div > div.inner-conntent > div:nth-child(3) > div > form > div:nth-child(1) > div > div > input')))
        password = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                                   '#base > div.content-outter > div > div.inner-conntent > div:nth-child(3) > div > form > div:nth-child(2) > div > div.ivu-input-wrapper.ivu-input-type.ivu-input-group.ivu-input-group-with-prepend > input')))
        email.send_keys(self.email)
        password.send_keys(self.password)

    def get_gap(self, image1, image2):
        """
        获取缺口偏移量
        :param image1: 不带缺口图片
        :param image2: 带缺口图片
        :return:
        """
        left = 60  # 初始横坐标 跳过左侧待拼合的滑块  直接找右边的滑块
        for i in range(left, image1.size[0]):
            for j in range(image1.size[1]):
                if not self.is_pixel_equal(image1, image2, i, j):  # 比较像素  false为不一样
                    left = i
                    return left
        return left

    def is_pixel_equal(self, image1, image2, x, y):
        """
        判断两个像素是否相同
        :param image1: 图片1
        :param image2: 图片2
        :param x: 位置x
        :param y: 位置y
        :return: 像素是否相同
        """
        # 取两个图片的像素点
        pixel1 = image1.load()[x, y]
        pixel2 = image2.load()[x, y]
        threshold = 60  # 比较的像素阈值
        if abs(pixel1[0] - pixel2[0]) < threshold and abs(pixel1[1] - pixel2[1]) < threshold and abs(
                pixel1[2] - pixel2[2]) < threshold:
            return True
        else:
            return False

    def get_track(self, distance):
        """
        根据偏移量获取移动轨迹
        :param distance: 偏移量
        :return: 移动轨迹
        """
        track = []  # 移动轨迹
        current = 0  # 当前位移
        mid = distance * 4 / 5  # 减速阈值  0.8*距离
        t = 0.2  # 计算间隔
        v = 0  # 初速度

        while current < distance:
            if current < mid:
                a = 2  # 加速度为正2
            else:
                a = -3  # 加速度为负3
            v0 = v  # 初速度v0
            v = v0 + a * t  # 当前速度v = v0 + at
            move = v0 * t + 1 / 2 * a * t * t  # 移动距离x = v0t + 1/2 * a * t^2
            current += move  # 当前位移
            track.append(round(move))  # 加入轨迹 每隔一段时间的位移   round四舍五入
        return track

    def move_to_gap(self, slider, track):
        """
        拖动滑块到缺口处
        :param slider: 滑块
        :param track: 轨迹
        :return:
        """
        ActionChains(self.browser).click_and_hold(slider).perform()  # 点击鼠标拖动滑块
        for x in track:
            ActionChains(self.browser).move_by_offset(xoffset=x, yoffset=0).perform()  # 拖动
        time.sleep(0.5)
        ActionChains(self.browser).release().perform()  # 松开鼠标

    def login(self):
        """
        登录
        :return: None
        """
        submit = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'login-btn')))
        submit.click()
        time.sleep(10)
        print('登录成功')

    def crack(self):
        self.open()  # 输入用户名和密码
        button = self.get_geetest_button()  # 获取初始验证按钮   点击认证
        button.click()  # 点击验证按钮
        image1 = self.get_geetest_image('captcha1.png')  # 获取验证码图片  无缺口
        slider = self.get_slider()  # 点按滑块 呼出缺口
        slider.click()
        image2 = self.get_geetest_image('captcha2.png')  # 获取带缺口的验证码图片 #有缺口
        gap = self.get_gap(image1, image2)  # 获取缺口位置  相对于横坐标图片的 偏移量
        print('缺口位置', gap)
        gap -= BORDER  # 减去缺口位移
        track = self.get_track(gap)  # 获取移动轨迹
        print('滑动轨迹', track)  # 是一个列表  每隔0.2s的位移
        self.move_to_gap(slider, track)  # 拖动滑块  每隔一小段时间0.2s拖动一次  可以视为动态拖动

        success = self.wait.until(
            EC.text_to_be_present_in_element((By.CLASS_NAME, 'geetest_success_radar_tip_content'), '验证成功'))
        print(success)

        # 失败后重试
        if not success:
            self.crack()
        else:
            self.login()


if __name__ == '__main__':
    crack = CrackGeetest()
    crack.crack()
