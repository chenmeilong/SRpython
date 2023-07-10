

import bluetooth

server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

server_sock.bind(("", bluetooth.PORT_ANY))
server_sock.listen(1)

port = server_sock.getsockname()[1]
print("listening on port {:d}".format(port))

uuid = "0x0000FFF4-0000-1000-8000-00805F9B34FB"  #79A5A5E0-62B0-BB20-4DDB-B451159BFE07   16位转128位0x0000xxxx-0000-1000-8000-00805F9B34FB
bluetooth.advertise_service(server_sock, "FooBar Service", uuid)

client_sock, address = server_sock.accept()
print("Accepted connection from ", address)

data = client_sock.recv(1024)
print("received [{:r}]".format(data))

client_sock.close()
server_sock.close()