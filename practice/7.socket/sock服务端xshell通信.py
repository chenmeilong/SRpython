import socket ,os,time
server = socket.socket()
server.bind(('localhost',9999) )

server.listen()

while True:
    conn, addr = server.accept()
    print("new conn:",addr)
    while True:
        print("等待新指令")
        data = conn.recv(1024)
        if not data:
            print("客户端已断开")
            break
        print("执行指令:",data)
        cmd_res = os.popen(data.decode()).read() #接受字符串，执行结果也是字符串   返回os命令
        print("before send",len(cmd_res))
        if len(cmd_res) ==0:
            cmd_res = "cmd has no output..."

        conn.send( str(len(cmd_res.encode())).encode("utf-8")    )     #先发大小给客户端
        time.sleep(0.5)                                      #sleep就会缓冲区超时两个分开发过去
        conn.send(cmd_res.encode("utf-8"))                  #连续两个send会 连包   缓冲区会把两次合成一次 发送过去 强制性
        print("send done")
        # os.path.isfile()
        # os.stat("sock")
server.close()

# import hashlib                            #加密模块  可以此模块加密
# m = hashlib.md5()