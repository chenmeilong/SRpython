import socketserver               #支持多个客户端   无限发送

class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):             #跟客服端所以的交互 都在这处理
        while True:
            try:                                                  #客户端断开时 抓异常处理
                self.data = self.request.recv(4096).strip()             #self是客户请求实例化
                print("{} wrote:".format(self.client_address[0]))       #打印ip地址
                print(self.data)
                self.request.send(self.data.upper())                    #大写传回
            except ConnectionResetError as e:                           #
                print("err",e)
                break
if __name__ == "__main__":
    HOST, PORT = '192.168.10.144', 8080
    # Create the server, binding to localhost on port 9999
    server = socketserver.ThreadingTCPServer((HOST, PORT), MyTCPHandler)        #传ip 传类      ThreadingTCPServer多线程
    server.serve_forever()

