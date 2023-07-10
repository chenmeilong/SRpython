class Flight(object):
    def __init__(self, name):
        self.flight_name = name

    def checking_status(self):
        print("checking flight %s status " % self.flight_name)
        return 1

    @property                                                                #方法变静态属性
    def flight_status(self):
        status = self.checking_status()
        if status == 0:
            print("flight got canceled...")
        elif status == 1:
            print("flight is arrived...")
        elif status == 2:
            print("flight has departured already...")
        else:
            print("cannot confirm the flight status...,please check later")

    @flight_status.setter                # 修改
    def flight_status(self, status):
        status_dic = {
            0: "canceled",
            1: "arrived",
            2: "departured"
        }
        print("\033[31;1mHas changed the flight status to \033[0m", status_dic.get(status))

    @flight_status.deleter                        # 删除    可以删除类变量等属性
    def flight_status(self):
        print("status got removed...")


f = Flight("CA980")
f.flight_status                                      #此时已经变成一个静态属性了，不是方法了，想调用已经不需要加()号了，直接f.flight_status
f.flight_status = 2                                  # 触发@flight_status.setter      修改
del f.flight_status                                  # 触发@flight_status.deleter     删除
