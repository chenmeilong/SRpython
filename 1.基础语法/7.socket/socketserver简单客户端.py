import socket

client = socket.socket() #声明socket类型，同时生成socket连接对象                    1声明实例  2连上  3发数据
client.connect(('v410842y77.wicp.vip',12177))         #本地9999端口              localhost可以换成ip地址，进行异地通信


while True:
    msg = input(">>:").strip()                          #输入数据转字符串
    if len(msg) == 0:continue                          #为空就从新输入，不能发空
    client.send(msg.encode("utf-8"))                    #转 二进制并发送
    data = client.recv(1024)                           #设置收多少字节
    print("recv:",data.decode())
client.close()

