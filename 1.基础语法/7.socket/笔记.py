# 断言
# assert type(obj.name) is int
# print(obj.name /2)
#
#
#     服务端
#     server = socket.socket(AF.INET,sock.SOCK_STREAM)               #声明
#     server.bind(localhost,9999)                                    #绑定ip地址和端口
#     server.listen()                                                #监听
#     while True:
#         conn,addr = server.accept() #卡在这等   阻塞                             #客户端发过来的一个地址    conn为新来的连接开一个实例
#         while True:
#            print("new conn",addr)
#            data = conn.recv(1024) #小于8192  即8k   #recv默认是阻塞的
#
#            if not data:
#                 break #客户端已断开，   如果没有这条语句 conn.recv收到的就都是空数据， 会进入死循环
#            print(data)
#            conn.send(data.upper())                                 #变大写写发回去
#     客户端
#        client = socket.socket()                                  #声明实例  连上  发数据
#        client.connect(serverip, 9999 )
#        client.send(data)
#        client.send(data)                                       #
#        client.recv(dat)
#
#     socket 粘包
#
#     ftp server      shell操作+
#     1. 读取文件名
#     2. 检测文件是否存在
#     3. 打开文件
#     4. 检测文件大小
#     5. 发送文件大小给客户端
#     6. 等客户端确认
#     7. 开始边读边发数据
#     8. 发送md5
#
#
# First, you must create a request handler处理类 class by subclassing the BaseRequestHandler class and overriding覆盖 its handle() method; this method will process incoming requests. 　　
# 1.你必须自己创建一个请求处理类，并且这个类要继承BaseRequestHandler,并且还有重写父亲类里的handle()
# Second, you must instantiate实例化 one of the server classes, passing it the server’s address and the request handler class.
# 2.你必须实例化TCPServer ，并且传递server ip 和 你上面创建的请求处理类 给这个TCPServer 实例
# Then call the handle_request() or serve_forever() method of the server object to process one or many requests.
# server.handle_request() #只处理一个请求 ，然后退出
# server.serve_forever() #处理多个一个请求，永远执行
#
#
# Finally, call server_close() to close the socket.
#
# chenronhua
#     a
#         a1
#         a2
#     b
#     c
# chen: cd a
# cd ..
#
# user_current_dir = "/home/chenronghua"
# user_current_dir = "/home/chenronghua"
#
#
#
#
# os.chdir