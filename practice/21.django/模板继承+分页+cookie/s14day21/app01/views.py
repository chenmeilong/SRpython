from django.shortcuts import render, HttpResponse,redirect
from django.urls import reverse

# Create your views here.

# def index(request):
#     # v = reverse('author:index')
#     # print(v)
#     from django.core.handlers.wsgi import WSGIRequest
#     # print(type(request))
#     # ��װ�������û�������Ϣ
#     # print(request.environ)
#     # for k,v in request.environ.items():
#     #     print(k,v)
#     # print(request.environ['HTTP_USER_AGENT'])
#     # request.POST
#     # request.GET
#     # request.COOKIES
#
#     return HttpResponse('OK')


def tpl1(request):
    user_list = [1, 2, 3, 43]
    return render(request, 'tpl1.html', {'u': user_list})


def tpl2(request):
    name = 'root'
    return render(request, 'tpl2.html', {'name': name})


def tpl3(request):
    status = "已经删除"
    return render(request, 'tpl3.html', {'status': status})

def tpl4(request):
    name = "IYMDFjfdf886sdf"
    return render(request, 'tpl4.html', {'name': name})


from  utils import pagination
LIST = []
for i in range(500):
    LIST.append(i)

def user_list(request):
    current_page = request.GET.get('p', 1)
    current_page = int(current_page)
    print(request.COOKIES)
    val = request.COOKIES.get('per_page_count',10)
    val = int(val)
    page_obj = pagination.Page(current_page,len(LIST),val)

    data = LIST[page_obj.start:page_obj.end]

    page_str = page_obj.page_str("/user_list/")

    return render(request, 'user_list.html', {'li': data,'page_str': page_str})




########################### cookie ###########################
user_info = {
    'dachengzi': {'pwd': "123123"},
    'kanbazi': {'pwd': "kkkkkkk"},
}
def login(request):
    if request.method == "GET":
        return render(request,'login.html')
    if request.method == "POST":
        u = request.POST.get('username')
        p = request.POST.get('pwd')
        dic = user_info.get(u)
        if not dic:
            return render(request,'login.html')
        if dic['pwd'] == p:
            res = redirect('/index/')
            # res.set_cookie('username111',u,max_age=10)
            # import datetime
            # current_date = datetime.datetime.utcnow()
            # current_date = current_date + datetime.timedelta(seconds=5)
            # res.set_cookie('username111',u,expires=current_date)
            res.set_cookie('username111',u)
            res.set_cookie('user_type',"asdfjalskdjf",httponly=True)
            return res
        else:
            return render(request,'login.html')

def auth(func):
    def inner(reqeust,*args,**kwargs):
        v = reqeust.COOKIES.get('username111')
        if not v:
            return redirect('/login/')
        return func(reqeust, *args,**kwargs)
    return inner
#FBV 装饰器 基于cookie的FBV
@auth
def index(reqeust):
    # 获取当前已经登录的用户
    v = reqeust.COOKIES.get('username111')
    return render(reqeust,'index.html',{'current_user': v})

from django import views
from django.utils.decorators import method_decorator

@method_decorator(auth,name='dispatch')
class Order(views.View):

    # @method_decorator(auth)
    # def dispatch(self, request, *args, **kwargs):
    #     return super(Order,self).dispatch(request, *args, **kwargs)

    # @method_decorator(auth)
    def get(self,reqeust):
        v = reqeust.COOKIES.get('username111')
        return render(reqeust,'index.html',{'current_user': v})

    def post(self,reqeust):
        v = reqeust.COOKIES.get('username111')
        return render(reqeust,'index.html',{'current_user': v})



def order(reqeust):
    # 获取当前已经登录的用户
    v = reqeust.COOKIES.get('username111')
    return render(reqeust,'index.html',{'current_user': v})

def cookie(request):
    #
    # request.COOKIES
    # request.COOKIES['username111']
    request.COOKIES.get('username111')

    response = render(request,'index.html')
    response = redirect('/index/')
    # 设置cookie，关闭浏览器失效
    response.set_cookie('key',"value")
    # 设置cookie, N秒只有失效
    response.set_cookie('username111',"value",max_age=10)
    # 设置cookie, 截止时间失效
    import datetime
    current_date = datetime.datetime.utcnow()
    current_date = current_date + datetime.timedelta(seconds=5)
    response.set_cookie('username111',"value",expires=current_date)
    response.set_cookie('username111',"value",max_age=10)

    # request.COOKIES.get('...')
    # response.set_cookie(...)
    obj = HttpResponse('s')

    obj.set_signed_cookie('username',"kangbazi",salt="asdfasdf")
    request.get_signed_cookie('username',salt="asdfasdf")

    return response























