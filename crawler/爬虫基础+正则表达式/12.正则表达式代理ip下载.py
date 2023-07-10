import urllib.request
import re

def open_url(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent',
                   'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.65 Safari/537.36')
    reponse = urllib.request.urlopen(req)
    html = reponse.read()
    return html

def get_ip(url):
    html = open_url(url).decode('utf-8')
    p = r'(([0,1]?\d?\d|2[0-4]\d|25[0-5])\.){3}([0,1]?\d?\d|2[0-4]\d|25[0-5])'
    #p = r'(?:(?:[0,1]?\d?\d|2[0-4]\d|25[0-5])\.){3}(?:[0,1]?\d?\d|2[0-4]\d|25[0-5])'     #(?:...) 表示非捕获组     通过这个优化
    iplist = re.findall(p, html)

    for each in iplist:
        print(each)

if __name__ == "__main__":
    url = "https://www.xicidaili.com/wt/"
    get_ip(url)