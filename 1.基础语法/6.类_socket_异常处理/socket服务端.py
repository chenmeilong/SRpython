#服务器端              只能支持一个连接
import socket
server = socket.socket()              #声明实例
server.bind(('192.168.10.114', 8080))    #绑定要监听端口
server.listen(5)                   #监听   最多允许5个连接挂起
print("我要开始等电话了")
while True:
    conn, addr = server.accept()  # 等电话打进来
    print(conn, addr)
    print("电话来了")
    conn.send(b'21')           #upper大写
    conn.close()



