from django.utils.deprecation import MiddlewareMixin

class Row1(MiddlewareMixin):
    def process_request(self,request):
        print('王森')

    def process_view(self, request, view_func, view_func_args, view_func_kwargs):
        print('张欣彤')

    def process_response(self, request, response):
        print('扛把子')
        return response

from django.shortcuts import HttpResponse
class Row2(MiddlewareMixin):
    def process_request(self,request):
        print('程毅强')
        # return HttpResponse('走')

    def process_view(self, request, view_func, view_func_args, view_func_kwargs):
        print('张需要')

    def process_response(self, request, response):
        print('侯雅凡')
        return response

class Row3(MiddlewareMixin):
    def process_request(self,request):
        print('刘东')

    def process_view(self, request, view_func, view_func_args, view_func_kwargs):
        print('邵林')

    def process_response(self, request, response):
        print('连之泪')
        return response

    def process_exception(self, request, exception):
        if isinstance(exception,ValueError):
            return HttpResponse('出现异常》。。')

    def process_template_response(self,request,response):
        # 如果Views中的函数返回的对象中，具有render方法
        print('-----------------------')
        return response