#python标准库提供的独立WSGI服务器称为wsgiref
from wsgiref.simple_server import make_server
def RunServer(environ, start_response):
    #environ  客户端发来所有数据
    #start_response  封裝要返回给用户的数据 ，响应头的状态
    start_response('200 OK', [('Content-Type', 'text/html')])
    #返回的内容
    return [bytes('<h1>Hello, web!</h1>', encoding='utf-8'), ]

if __name__ == '__main__':
    httpd = make_server('', 8000, RunServer)
    print("Serving HTTP on port 8000...")
    httpd.serve_forever()