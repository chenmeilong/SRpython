#客户端
import socket

client = socket.socket() #声明socket类型，同时生成socket连接对象                    1声明实例  2连上  3发数据
client.connect(('localhost',8080))         #本地6969端口

#简单例子
# client.send(b"helloword")
data = client.recv(1024)  # 设置收多少字节  多余的存入缓冲区 依次发送
print("recv:", data)      #接收到的大写回来的数据
client.close()

