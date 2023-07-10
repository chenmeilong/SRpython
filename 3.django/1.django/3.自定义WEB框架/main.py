#!/usr/bin/env python
# -*- coding:utf-8 -*-

from wsgiref.simple_server import make_server
from Controller import account

URL_DICT = {
     '/index':account.index,
     '/login':account.login,
}

def run_server(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    url = environ['PATH_INFO']
    func = None
    if url in URL_DICT:
         func = URL_DICT[url]
    if func:
        return func()
    else:
        return ['<h1>404</h1>'.encode('utf-8'),]

if __name__ == '__main__':
    httpd = make_server('', 8000, run_server)
    print ("Serving HTTP on port 8000...")
    httpd.serve_forever()