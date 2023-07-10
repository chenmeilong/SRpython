from django.shortcuts import render,redirect,HttpResponse

def login(request):
    # from django.conf import settings
    # print(settings.CSRF_HEADER_NAME)
    # HTTP_X_CSRFTOKEN
    # X-CSRFtoken
    if request.method == "GET":
        return render(request,'login.html')
    elif request.method == "POST":
        user = request.POST.get('user')
        pwd = request.POST.get('pwd')
        if user == 'root' and pwd == "123":
            # session中设置值
            request.session['username'] = user
            request.session['is_login'] = True
            if request.POST.get('rmb',None) == '1':
                # 超时时间
                request.session.set_expiry(10)
            return redirect('/index/')
        else:
            return render(request,'login.html')


from django.views.decorators.csrf import csrf_exempt,csrf_protect

@csrf_protect
def index(request):
    # session中获取值
    if request.session.get('is_login',None):
        return render(request,'index.html',{'username': request.session['username']})
    else:
        return HttpResponse('gun')

def logout(request):
    # del request.session['username']
    request.session.clear()
    return redirect('/login/')

#
class Foo:
    def __init__(self,req,html,dic):
        self.req = req
        self.html = html
        self.dic = dic
    def render(self):
        # // 创建钩子
        return render(self.req,self.html,self.dic)

def test(request,nid):
    print('小姨妈-->没带钱')
    # return render(request, 'index.html', {...})
    return Foo(request, 'index.html', {'k1': 'v1'})

from django.views.decorators.cache import cache_page

@cache_page(10)
def cache(request):
    import time
    ctime = time.time()
    return render(request, 'cache.html', {'ctime': ctime})

def signal(reuqest):
    from app01 import models

    obj = models.UserInf(user='root')
    print('end')
    obj.save()

    obj = models.UserInf(user='root')
    obj.save()

    obj = models.UserInf(user='root')
    obj.save()

    from sg import pizza_done


    pizza_done.send(sender="asdfasdf",toppings=123, size=456)


    return HttpResponse('ok')





######################## Form #####################
from django import forms
from django.forms import widgets
from django.forms import fields
class FM(forms.Form):
    # 字段本身只做验证
    user = fields.CharField(
        error_messages={'required': '用户名不能为空.'},
        widget=widgets.Textarea(attrs={'class': 'c1'}),
        label="用户名",
        )
    pwd = fields.CharField(
        max_length=12,
        min_length=6,
        error_messages={'required': '密码不能为空.', 'min_length': '密码长度不能小于6', "max_length": '密码长度不能大于12'},
        widget=widgets.PasswordInput(attrs={'class': 'c2'})
    )
    email = fields.EmailField(error_messages={'required': '邮箱不能为空.','invalid':"邮箱格式错误"})

    f = fields.FileField()

    # p = fields.FilePathField(path='app01')

    city1 = fields.ChoiceField(
        choices=[(0,'上海'),(1,'广州'),(2,'东莞')]
    )
    city2 = fields.MultipleChoiceField(
        choices=[(0,'上海'),(1,'广州'),(2,'东莞')]
    )

from app01 import models
def fm(request):
    if request.method == "GET":
        # 从数据库中吧数据获取到
        dic = {
            "user": 'r1',
            'pwd': '123123',
            'email': 'sdfsd',
            'city1': 1,
            'city2': [1,2]
        }
        obj = FM(initial=dic)
        return render(request,'fm.html',{'obj': obj})
    elif request.method == "POST":
        # 获取用户所有数据
        # 每条数据请求的验证
        # 成功：获取所有的正确的信息
        # 失败：显示错误信息
        obj = FM(request.POST)
        r1 = obj.is_valid()
        if r1:
            # obj.cleaned_data
            models.UserInf.objects.create(**obj.cleaned_data)
        else:
            # ErrorDict
            # print(obj.errors.as_json())
            # print(obj.errors['user'][0])
            return render(request,'fm.html', {'obj': obj})
        return render(request,'fm.html')











