#RFCOMM方式进行通信
import bluetooth

server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

port = 1
server_sock.bind(("A3:3A:11:0A:23:79", port))    #mac 地址
server_sock.listen(1)

client_sock, address = server_sock.accept()
print("Accepted connection from ", address)    #A3:3A:11:0A:23:79


data = client_sock.recv(1024)
print("received [%s]" % data)

client_sock.close()
server_sock.close()