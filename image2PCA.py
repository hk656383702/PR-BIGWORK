"""
2021/7/5
程序功能：
①将原先提取到的image矩阵（原始图像矩阵）（1200*255*255）进行PCA特征提取，提取过后的维度为（1200*150）
②将原先提取到的HOGfeature矩阵（1200*69192）进行PCA特征提取，提取过后的维度为（1200*150）

"""
import numpy as np
from sklearn.decomposition import PCA

img=np.load('image.npy')
imgtest=np.zeros([1200,img[1, :, :].reshape(1,-1).shape[1]])
HOG_features = np.load('HOG_feature.npy')
for i in range(1200): #将灰度图片转化为1200×N的维度
    epoch = i
    imgtest[i,:]=img[epoch, :, :].reshape(1, -1)


pca = PCA(n_components=150)
newX=[]
pca.fit(imgtest)
a=pca.fit_transform(imgtest)
pca.fit(HOG_features)
b=pca.fit_transform(HOG_features)
print(a.shape)
np.save('PCAfeature.npy', a)
np.save('HOG_AND_PCA_feature', b)