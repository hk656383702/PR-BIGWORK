"""
2021/7/5
程序功能：
①该程序将所有的1200副的灰度图像，通过HOG进行特征提取
②提取到的特征维度是1200*69192
问题1：1200*69192的维度是如何来的呢？
答：通过调整参数orientations=18, pixels_per_cell=[8,8], cells_per_block=[2, 2],
orientations=n：表示将180°的梯度方向平均分成n个区间
pixels_per_cell=[a,b]： 表示cell的大小为a×b
cells_per_block=[c, d]: 表示每个cell将被分成m行n列个子区域
具体的怎么算出69192，跟上述的三个参数+原始矩阵维度有关。
设提取到的特征维度为y,原始图像维度为k1,k2，y=f(n,a,b,c,d,k1,k2)
问题2：为什么要进行HOG特征提取？
答：HOG特征提取能够描述图像小范围的变化，俗称图像的细节，而通过眼底图像对糖尿病患者判断是否得眼部病变，需要医生观察血管，眼底出血，黄斑大小等细节，觉得HOG能够对分类有帮助
"""
import numpy as np
import xlrd
from skimage import feature as ft
import matplotlib.pyplot as plt
img=np.load('image.npy')
print(img.shape)
A=np.zeros((1200,69192),dtype=np.float32)
for i in range(1200):
    features = ft.hog(img[i], orientations=18, pixels_per_cell=[8,8], cells_per_block=[2, 2], visualize=True)
    A[i, :]=features[0]
    if i==0:
        plt.subplot(1,2,1)
        plt.imshow(features[1])
        plt.subplot(1,2,2)
        plt.plot(range(69192), list(features[0]))
        plt.show()

    print(f'{i}')
# np.save('HOG_feature.npy', A)
