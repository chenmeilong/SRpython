#-*-coding:gbk-*-                               #�ļ�������Ҫ����ʲô�ַ���ʽ��дʲô�ַ�������Ĭ��Ϊ�����utf8 ����Ҫ����
                                                #���ڿ��������ݶ���unicode��ʽ����
import sys
print(sys.getdefaultencoding())
__author__ = "Alex Li"

s = "���"                                          #��ʵ����Ļ���unicode
print(s.encode("gbk"))                               #gbk����      ����unicodeת����gbk
print(s.encode("utf-8"))                              #utf8����
print(s.encode("utf-8").decode("utf-8").encode("gb2312").decode("gb2312"))
