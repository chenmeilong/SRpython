import bluetooth

#通过用户友好的名字来寻找通信对象

target_name = "iphone"           #能找到我的手机
target_address = None

nearby_devices = bluetooth.discover_devices()      #当前环境的所有蓝牙设备  扫描

print(nearby_devices)


for bdaddr in nearby_devices:
    if target_name == bluetooth.lookup_name(bdaddr):     # 获取蓝牙的名字 扎到蓝牙的名字对应的地址
        target_address = bdaddr
        break

if target_address is not None:
    print("found target bluetooth device with address ", target_address)
else:
    print("could not find target bluetooth device nearby")


