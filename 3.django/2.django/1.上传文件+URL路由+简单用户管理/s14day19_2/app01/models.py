from django.db import models

#重新创建时 需要删除0001.initial.py文件

# Create your models here.
# app01_userinfo
class UserInfo(models.Model):
    # id列，自增，主键
    # 用户名列，字符串类型，指定长度
    username = models.CharField(max_length=32)
    password = models.CharField(max_length=64)

