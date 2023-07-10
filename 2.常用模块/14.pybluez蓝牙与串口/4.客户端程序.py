#RFCOMM方式进行通信
import bluetooth
bd_addr = "A3:3A:11:0A:23:79"     #服务端地址   A3:3A:11:0A:23:79
port = 1
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((bd_addr, port))      #连接到指定蓝牙设备

sock.send("hello!!")

sock.close()