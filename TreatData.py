# -*- coding: utf-8 -*-
"""
2021/7/5
函数编写人：黄凯
程序功能：
①该程序将所有的1200副图，转化为维度为1200*255*255的灰度图像
②汇总所有标签信息，取消注释，稍微修改即可
问题1：原本是2240*1488的图像为什么提取后是255*255呢？
为了减少图片占用的存储空间，采用resize函数，将原本2240*1488的图像放缩至255*255，resize函数能尽可能的保留图像的原始信息

"""

import os
import cv2
import numpy as np
import xlrd



#%%
n = 1200
name = []
xlsname = []
init_label = []
label = [0]*n
label4 = [0]*n
label3 = [0]*n
labelRetinopathy = [0]*n
labelMacular_edema = [0]*n
for i in range(12):
    name = name + os.listdir(f'./images/Base{i}')
    xlsname.append((i+1)*100+i)
for i in xlsname[::-1]:
    name.pop(i)

x = np.zeros((n, 256, 256), dtype=np.float32)
y = np.zeros((n), dtype=np.int64)
for i in range(12):
    for j in range(100):
        epoch = i*100+j
        try:
            img = cv2.imread(f'./images/Base{i}/{name[epoch]}')
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.imread(f'./images/Base{i}/{name[epoch]}', cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            x[epoch, :, :] = img
        except:
            print(f'Error in {epoch}')
            break
    print(f'Folder {i}')
np.save('image.npy', x)
#%%
#
labellist = os.listdir('./annotation of images')
for i in labellist:
    labelxls = xlrd.open_workbook('./annotation of images/'+i)
    sheet = labelxls.sheet_by_index(0)#索引的方式，从0开始
    for i in range(1,101):
        init_label.append(sheet.row_values(i))

for i in range(n):
    if init_label[i][2] == 0:
        labelRetinopathy[i] = 0
    else:
        labelRetinopathy[i] = 1
    if init_label[i][3] == 0:
        labelMacular_edema[i] = 0
    else:
        labelMacular_edema[i] = 1

for i in range(n):
    label4[i] = init_label[i][2]

for i in range(n):
    label3[i] = init_label[i][3]


np.save('label4.npy', label4)                           #视网膜病变四分类标签
np.save('label3.npy', label3)                           #黄斑病变三分类标签
np.save('labelRetinopathy.npy',labelRetinopathy)        #视网膜病变二分类标签
np.save('labelMacular_edema.npy', labelMacular_edema)   #黄斑病变二分类标签