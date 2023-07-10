#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2021/3/29 21:18
#@Author: Ang
#@File  : auto_web1.py
import requests
import time
import datetime
import os
from selenium import webdriver


from selenium.webdriver.common.keys import Keys
# 环境配置
def auto_web_1():
    chromedriver = "C:/Program Files/Google/Chrome/Application"
    os.environ["webdriver.ie.driver"] = chromedriver

    driver = webdriver.Chrome()  # 选择Chrome浏览器
    #  3月29日3844068    3月30日 3847110
    driver.get('http://one.hrbeu.edu.cn')
    driver.maximize_window()  # 最大化谷歌浏览器
    time.sleep(10)
##############################################################################
##############################################################################
###########################################################################
    username = "S320040261"             # 请替换成你的用户名   S
    password = "xxxxxxxxxxxxx"                       # 请替换成你的密码
    user_phone = "15108215920"              #个人手机号
    user_local = "15"                       #公寓
    user_teacher = "冯赞元"                #导员姓名
    user_teacher_number = "19845255393"    #导员号码
    user_thing1 = "水果街吃饭"           #报备理由
    user_thing2 = "水果街"                    #外出地点
    user_time_begin = "06:02"             #时间1
    user_time_end = "21:58"                #时间2
##############################################################################
##############################################################################
##############################################################################
    driver.find_element_by_id('username').click()  # 点击用户名输入框
    print(driver.find_element_by_id('username').click())
    driver.find_element_by_id('username').clear()  # 清空输入框
    driver.find_element_by_id('username').send_keys(username)  # 自动敲入用户名
    time.sleep(2)
    driver.find_element_by_id('password').click()  # 点击密码输入框
    driver.find_element_by_id('password').clear()  # 清空输入框
    driver.find_element_by_id('password').send_keys(password)  # 自动敲入密码

    # 采用class定位登陆按钮
    # driver.find_element_by_id('ext-gen29').click()
    driver.find_element_by_id('login-submit').click()  # 点击登录
    # driver.find_element_by_class_name('ant-btn').click() # 点击“登录”按钮
    # 采用xpath定位登陆按钮，
    # driver.find_element_by_xpath('//*[@id="root"]/div/div[3]/form/button').click()
    #/html/body/div[2]/div/div[2]/div[12]/div[2]/div/p
    #/html/body/div[2]/div/div[2]/div[12]/div[2]/ul/li[2]/a[1]
    time.sleep(10)
    driver.find_element_by_xpath('/html/body/div[2]/div/div[2]/div[12]/div[2]/div/p').click()  # 电机更多
    time.sleep(3)

    driver.find_element_by_xpath('/html/body/div[2]/div/div[2]/div[12]/div[2]/ul/li[2]/a[1]').click()  # 进出线报备
    time.sleep(10)
    windows = driver.window_handles
    driver.switch_to.window(windows[-1])
    driver.find_element_by_id('V1_CTRL5').click()  # 点击电话输入框
    driver.find_element_by_id('V1_CTRL5').clear()  # 清空输入框
    driver.find_element_by_id('V1_CTRL5').send_keys(user_phone)  # 自动敲入电话
    # time.sleep(1)
    driver.find_element_by_id('V1_CTRL6').click()  # 点击公寓输入框
    driver.find_element_by_id('V1_CTRL6').clear()  # 清空输入框
    driver.find_element_by_id('V1_CTRL6').send_keys(user_local)  # 自动敲入公寓

    
    driver.find_element_by_class_name("selection").click()  # 选择导员姓名输入
    time.sleep(2)
    driver.find_element_by_class_name("select2-search__field").send_keys(user_teacher)
    time.sleep(2)

 
    driver.find_element_by_class_name("select2-search__field").send_keys(Keys.ENTER)
    driver.find_element_by_id('V1_CTRL8').click()  # 点击导员电话
    driver.find_element_by_id('V1_CTRL8').clear()  # 清空输入框
    driver.find_element_by_id('V1_CTRL8').send_keys(user_teacher_number)  # 自动敲入
    time.sleep(1)

    driver.find_element_by_id('V1_CTRL9').click()  # 点击进出校报备输入框
    time.sleep(1)
    driver.find_element_by_id('V1_CTRL11').click()  # 点击出校事由及行程安排输入框
    driver.find_element_by_id('V1_CTRL11').clear()  # 清空输入框
    driver.find_element_by_id('V1_CTRL11').send_keys(user_thing1)  # 自动敲入
    time.sleep(1)
    driver.find_element_by_id('V1_CTRL18').click()  # 点击  出校类别其他
    time.sleep(1)
    driver.find_element_by_id('V1_CTRL20').click()  # 点击开始时间
    driver.find_element_by_id('V1_CTRL20').clear()  # 清空输入框
    driver.find_element_by_id('V1_CTRL20').send_keys(user_time_begin)  # 自动敲入
    time.sleep(1)
    driver.find_element_by_id('V1_CTRL22').click()  # 点击结束时间
    driver.find_element_by_id('V1_CTRL22').clear()  # 清空输入框
    driver.find_element_by_id('V1_CTRL22').send_keys(user_time_end)  # 自动敲入
    time.sleep(1)

    driver.find_element_by_id('V1_CTRL6').click()  # 点击公寓输入框
    driver.find_element_by_id('V1_CTRL26').click()  # 点击目的地
    driver.find_element_by_id('V1_CTRL26').clear()  # 清空输入框
    driver.find_element_by_id('V1_CTRL26').send_keys(user_thing2)  # 自动敲入

    driver.find_element_by_class_name("command_button_content").click()
    time.sleep(10)
    
    driver.find_element_by_xpath('/html/body/div[9]/div/div[2]/button[1]').click()
    time.sleep(10)
    driver.find_element_by_xpath('/html/body/div[10]/div/div[2]/button').click()
    ############################################################################################################
    #替换token
    requests.get('http://pushplus.hxtrip.com/send?token=xxxxxxxxxxxx'+'&title=自动报备已完成!'+'&content=尊敬的陈先生，您好！'+str(datetime.datetime.now().strftime('%Y-%m-%d')) +'自动报备已完成。')
############################################################################################################
    time.sleep(10)
    driver.quit()
    #print("自动报备已完成")
    

auto_web_1()

