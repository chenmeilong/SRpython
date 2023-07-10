#读操作  调试完成
# import xlrd
# data = xlrd.open_workbook('readtest.xls')      #打开Excel文件读取数据
# table = data.sheet_by_name(u'5000条抖音用户数据')   #通过名称获取工作表
# print( table.nrows)             #行数
# print(table.ncols)             #列数
# print(table.row_values(11))     #列表输出行
# print( table.col_values(1))     #列表输出行
# print(table.cell(0,0).value)    #获取单元格

#写操作    调试完成
# import xlwt
# workbook = xlwt.Workbook(encoding = 'ascii')
# worksheet = workbook.add_sheet('My Worksheet')
# worksheet.write(0, 1, label='中文试试Column 0 Value')                   #支持中文，，第一行  第二列  会将原来数据的擦除
# worksheet.write(1, 1, label='中文试试Column 0 Value')
# workbook.save('Excel_Workbook.xls')

#改操作 不修改原数据的基础上修改
# import xlrd
# from xlutils.copy import copy
# book = xlrd.open_workbook("Excel_Workbook.xls")      #打开文件
# nr = book.sheet_by_index(0).nrows                     #打开工作表的行数
# book_copy = copy(book)                                #复制原来的文件
# sheet = book_copy.get_sheet(0)                        #打开复制的工作表
# u = [3,"aqiuqiu","123",500,"jh"]
# for i in range(u.__len__()):
#     sheet.write(nr,i,u[i])                                #在nr行往后写i个数据，数据在u[i]中
# book_copy.save("Excel_Workbook.xls")




