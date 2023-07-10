
def index():
    # return 'index'
    import time
    v=str(time.time())
    f = open('Views/index.html',mode='rb')
    data = f.read()
    f.close()
    data=data.replace(b'@uuuuu',v.encode('utf-8'))
    return [data,]

def login():
    # return 'login'
    f = open('Views/login.html',mode='rb')
    data = f.read()
    f.close()
    return [data,]

