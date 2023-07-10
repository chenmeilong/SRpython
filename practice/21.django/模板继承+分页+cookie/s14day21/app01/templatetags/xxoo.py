__author__ = 'Administrator'
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.simple_tag
def houyafan(a1,a2,a3):
    return a1 + a2

@register.filter
def jiajingze(a1,a2):
    print(a2,type(a2))
    return a1 + str(a2)