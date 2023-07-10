from django.shortcuts import render,HttpResponse

# Create your views here.


def login2(request):
    if request.method == "GET":
        return render(request, 'login2.html')
    elif request.method == "POST":
        # radio
        # v = request.POST.get('gender')
        # print(v)
        # v = request.POST.getlist('favor')
        # print(v)
        # v = request.POST.get('fafafa')
        # print(v)
        obj = request.FILES.get('fafafa')
        print(obj,type(obj),obj.name)
        import os
        file_path = os.path.join('upload', obj.name)
        f = open(file_path, mode="wb")
        for i in obj.chunks():
            f.write(i)
        f.close()

        from django.core.files.uploadedfile import InMemoryUploadedFile
        return render(request, 'login2.html')
    else:
        # PUT,DELETE,HEAD,OPTION...
        return HttpResponse('Home')