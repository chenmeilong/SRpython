from django.db import models

# Create your models here.

class UserInf(models.Model):

    user = models.CharField(max_length=32)