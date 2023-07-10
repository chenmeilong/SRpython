
from selenium import webdriver
import requests

url = 'https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gb18030&word=%D5%DB%B5%FE%BF%D5%BC%E4&fr=ala&ala=1&alatpl=adress&pos=0&hs=2&xthttps=111111'

driver = webdriver.Firefox()
driver.set_window_size(1000, 30000)
driver.get(url)
html = driver.find_element_by_xpath("//*").get_attribute("outerHTML")
print(html)