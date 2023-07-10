import random

arr = [random.randint(0,20) for i in range(0,10)]
print(arr)
def MergeSort2(arr, n):
    if n>0:
        i = 1
        while i<=n : #先从1开始，每次增加i。即归并块
            j=0
            while j<n-i:
                merge(arr, j, j+i-1, min(j+i+i-1, n-1))
                j += 2*i
            i+=i
MergeSort2(arr,10)
print(arr)