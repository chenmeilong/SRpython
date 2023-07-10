import sys
import socket
import time
import gevent
#不能收空数据
from gevent import socket, monkey

monkey.patch_all()    #把当前程序的所有的io操作给我单独的做上标记


def server(port):
    s = socket.socket()
    s.bind(('0.0.0.0', port))
    s.listen(500)
    while True:
        cli, addr = s.accept()
        gevent.spawn(handle_request, cli)     #启动协程


def handle_request(conn):
    try:
        while True:
            data = conn.recv(1024)
            print("recv:", data)
            conn.send(data)
            if not data:
                conn.shutdown(socket.SHUT_WR)

    except Exception as  ex:
        print(ex)
    finally:
        conn.close()


if __name__ == '__main__':
    server(8001)