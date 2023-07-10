import urllib.request
from bs4 import BeautifulSoup
import re

url = "http://baike.baidu.com/view/284853.htm"
response = urllib.request.urlopen(url)
html = response.read()
soup =  BeautifulSoup (html,"html.parser")

for each in soup.find_all(href=re.compile("view")):
    print (each.txt,"->",''.join(["http://baike.baidu.com",each["href"]]))

