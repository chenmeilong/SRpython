import bluetooth

nearby_devices = bluetooth.discover_devices(lookup_names=True)   #扫描蓝牙设备 返回 [('EC:D0:9F:CA:77:5C', 'myphone')]

print(nearby_devices)


for addr, name in nearby_devices:
    print("  %s - %s" % (addr, name))

    services = bluetooth.find_service(address=addr)
    for svc in services:
        print("Service Name: %s" % svc["name"])
        print("    Host:        %s" % svc["host"])
        print("    Description: %s" % svc["description"])
        print("    Provided By: %s" % svc["provider"])
        print("    Protocol:    %s" % svc["protocol"])
        print("    channel/PSM: %s" % svc["port"])
        print("    svc classes: %s " % svc["service-classes"])
        print("    profiles:    %s " % svc["profiles"])
        print("    service id:  %s " % svc["service-id"])
        print("")