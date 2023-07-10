"""s14day21 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url,include
from django.contrib import admin
from app01 import views
urlpatterns = [
    # url(r'^admin/', admin.site.urls),
    # url(r'^index/', views.index),
    # url(r'^index/', views.index, {'name': 'root'}),
    # url(r'^a/', include('app01.urls', namespace='author')),
    #tpl1-tpl4 模板操作 自定义函数 simple_tag 和filter
    url(r'^tpl1/', views.tpl1),
    url(r'^tpl2/', views.tpl2),
    url(r'^tpl3/', views.tpl3),
    url(r'^tpl4/', views.tpl4),
    url(r'^user_list/', views.user_list),
    url(r'^login/', views.login),
    url(r'^index/', views.index),
    url(r'^order/', views.Order.as_view()),
]
