from django.shortcuts import render,HttpResponse,redirect

def login(request):
    if request.method == "GET":
        return render(request, 'login.html')
    elif request.method == "POST":
        # 数据库中执行 select * from user where usernam='x' and password='x'
        u = request.POST.get('user')
        p = request.POST.get('pwd')
        # obj = models.UserInfo.objects.filter(username=u,password=p).first()
        # print(obj)# obj None,
        # count = models.UserInfo.objects.filter(username=u, password=p).count()
        obj = models.UserInfo.objects.filter(username=u, password=p).first()
        if obj:
            return redirect('/cmdb/index/')
        else:
            return render(request, 'login.html')
    else:
        # PUT,DELETE,HEAD,OPTION...
        return redirect('/index/')

def bim(request):
    if request.method == "GET":
        return render(request, 'bim.html')




def index(request):
    return render(request, 'index.html')

def user_info(request):
    if request.method == "GET":
        user_list = models.UserInfo.objects.all()
        # print(user_list.query)
        # QuerySet [obj,obj,]
        return render(request, 'user_info.html', {'user_list': user_list})
    elif request.method == 'POST':
        u = request.POST.get('user')
        p = request.POST.get('pwd')
        models.UserInfo.objects.create(username=u,password=p)
        return redirect('/cmdb/user_info/')
        # user_list = models.UserInfo.objects.all()
        # return render(request, 'user_info.html', {'user_list': user_list})

def user_detail(request, nid):
    obj = models.UserInfo.objects.filter(id=nid).first()
    # 去单挑数据，如果不存在，直接报错
    # models.UserInfo.objects.get(id=nid)
    return render(request, 'user_detail.html', {'obj': obj})

def user_del(request, nid):
    models.UserInfo.objects.filter(id=nid).delete()
    return redirect('/cmdb/user_info/')

def user_edit(request, nid):
    if request.method == "GET":
        obj = models.UserInfo.objects.filter(id=nid).first()
        return render(request, 'user_edit.html',{'obj': obj})
    elif request.method == "POST":
        nid = request.POST.get('id')
        u = request.POST.get('username')
        p = request.POST.get('password')
        models.UserInfo.objects.filter(id=nid).update(username=u,password=p)
        return redirect('/cmdb/user_info/')

from app01 import models
def orm(request):
    # 创建
    # 创建
    # models.UserInfo.objects.create(username='root',password='123')

    # dic = {'username': 'eric', 'password': '666'}
    # models.UserInfo.objects.create(**dic)

    # obj = models.UserInfo(username='alex',password='123')
    # obj.save()

    # 查
    # result = models.UserInfo.objects.all()
    # result = models.UserInfo.objects.filter(username='root',password='123')
    #
    # result,QuerySet => Django => []
    # [obj(id,username,password),obj(id,username,password), obj(id,username,password)]
    # for row in result:
    #     print(row.id,row.username,row.password)
    # print(result)

    # 删除
    # models.UserInfo.objects.filter(username="alex").delete()

    # 更新
    # models.UserInfo.objects.filter(id=3).update(password="69")

    return HttpResponse('orm')







# def home(request):
#     return HttpResponse('Home')
from django.views import View
class Home(View):

    def dispatch(self, request, *args, **kwargs):
        # 调用父类中的dispatch
        print('before')
        result = super(Home,self).dispatch(request, *args, **kwargs)
        print('after')
        return result

    def get(self,request):
        print(request.method)
        return render(request, 'home.html')

    def post(self,request):
        print(request.method,'POST')
        return render(request, 'home.html')

