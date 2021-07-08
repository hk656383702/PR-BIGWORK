"""
2021/7/6
程序功能：
①将提取到的特征直接投入到分类器
②画图
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 目的是通过检测眼底图像来判断病人的视网膜是否病变 0表示无病变 1表示病变等级1 2表示等级2 3表示等级3
# 记录在PCA中n值的不同，其准确率的变化的
record1 = []  # KNN
record2 = []  # NB
record3 = []  # DT
record4 = []  # SVM
record5 = []  # RandomForest
record6 = []  # BPNN
##控制量设置
# featureset==1 表示选取老师提供的特征集
# featureset==2 表示选取HOG通过oversample后的特征集
# featureset==3 表示选取PCA 通过oversample后的特征集
featureset = 2
test_size = 0.3
min_n_components = 10
max_n_components = 200
h_n_components = 10
oversample_method=1 #0表示随机过采样，1表示smote过采样
for i in range(min_n_components, max_n_components, h_n_components):
    n_components = i
    #####特征提取
    # 导入数据
    imgtest = np.load('image2GRAY.npy')  # 导入灰度图像数据
    HOG_features = np.load('HOG_feature.npy')  # 导入从灰度图像中HOG提取后的特征
    label = np.load('label4.npy')  # 导入标签0表示无病变 1表示病变等级1 2表示等级2 3表示等级3
    pca = PCA(n_components=n_components)  # 构建PCA类,提取n_components个特征
    pca.fit(imgtest)  # 对灰度图像数据进行PCA特征提取
    PCA_feature = pca.fit_transform(imgtest)
    pca.fit(HOG_features)  # 对HOG特征数据进行PCA特征提取
    HOG_PCA_feature = pca.fit_transform(HOG_features)
    y = label  # 导入视网膜是否病变标签 0表示无病变 1表示病变等级1 2表示等级2 3表示等级3
    print('完成PCA特征提取')

    ##过采样
    # 由于原始数据集正样本和负样本不均，因此采用过采样，使所有类样本均匀
    if oversample_method == 0:
        ros = RandomOverSampler(random_state=0)
        HOG_PCA_feature_oversampled, HOG_PCA_label_oversampled = ros.fit_resample(HOG_PCA_feature, y)
        PCA_feature_oversampled, PCA_label_oversampled = ros.fit_resample(PCA_feature, y)
    elif oversample_method == 1:
        HOG_PCA_feature_oversampled, HOG_PCA_label_oversampled = SMOTE().fit_resample(HOG_PCA_feature, y)
        PCA_feature_oversampled, PCA_label_oversampled = SMOTE().fit_resample(PCA_feature, y)
    print('完成过采样')

    if featureset == 1:  # featureset==1 表示选取老师提供的特征集
        X = np.load('teacher_feature.npy')
        y = np.load('DRlabel.npy')  # 糖尿病标签    1表示有糖尿病，0表示无糖尿病
    elif featureset == 2:  # featureset==3 表示选取HOG通过oversample后的特征集
        X = HOG_PCA_feature_oversampled
        y = HOG_PCA_label_oversampled
    elif featureset == 3:  # featureset==4 表示选取PCA 通过oversample后的特征集
        X = PCA_feature_oversampled
        y = PCA_label_oversampled

    ##数据集训练集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    # 数据标准化，给BP神经网络使用
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 建立分类器
    models = []
    names = []
    results = []
    num_folds = 10  ##10折交叉验证

    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('DT', DecisionTreeClassifier(criterion="entropy")))
    models.append(('SVM', SVC(kernel='poly')))
    models.append(('RandomForest', RandomForestClassifier(n_estimators=100)))
    models.append(('BPNN',MLPClassifier(hidden_layer_sizes=(5, 2), random_state=1,max_iter=10000)))
    tips = "'当前PCA提取的特征维度：'%d " % (n_components)
    print(tips)
    for name, model in models:
        kfold = model_selection.KFold(n_splits=num_folds)
        cv_results = model_selection.cross_val_score(model, X, y, n_jobs=-1, cv=kfold, scoring='accuracy')
        # AUC_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
        results.append(cv_results)
        names.append(name)
        y_pred = model.fit(X_train, y_train).predict(X_test)
        print("%s: 混淆矩阵为 \n%s" % (name, confusion_matrix(y_test, y_pred)))
        # msg = "%s: 'accuracy':%f (%f) 'AUC': %f (%f)" % (name, cv_results.mean(), cv_results.std(), AUC_results.mean(), AUC_results.std())
        msg = "%s: 'accuracy':%f (%f)" % (name, cv_results.mean(), cv_results.std())
        if name == 'KNN':
            record1.append(cv_results.mean())
        elif name == 'NB':
            record2.append(cv_results.mean())
        elif name == 'DT':
            record3.append(cv_results.mean())
        elif name == 'SVM':
            record4.append(cv_results.mean())
        elif name == 'RandomForest':
            record5.append(cv_results.mean())
        elif name == 'BPNN':
            record6.append(cv_results.mean())
        print(msg)
    # clf = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=1,max_iter=10000)
    # clf.fit(X_train_std, y_train)
    # sc = clf.score(X_test_std, y_test)
    # record6.append()
    # print('bp神经网络预测结果：', sc)

# 画图模块
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
fig = plt.figure(1)
ax1 = plt.subplot(2, 3, 1)
plt.plot(range(min_n_components, max_n_components, h_n_components), record1)
plt.xlabel('PCA特征选取数目')
plt.ylabel('KNN-10折交叉验证平均准确率')
ax2 = plt.subplot(2, 3, 2)
plt.plot(range(min_n_components, max_n_components, h_n_components), record2)
plt.xlabel('PCA特征选取数目')
plt.ylabel('NB-10折交叉验证平均准确率')
ax3 = plt.subplot(2, 3, 3)
plt.plot(range(min_n_components, max_n_components, h_n_components), record3)
plt.xlabel('PCA特征选取数目')
plt.ylabel('DT-10折交叉验证平均准确率')
ax4 = plt.subplot(2, 3, 4)
plt.plot(range(min_n_components, max_n_components, h_n_components), record4)
plt.xlabel('PCA特征选取数目')
plt.ylabel('SVM-10折交叉验证平均准确率')
ax5 = plt.subplot(2, 3, 5)
plt.plot(range(min_n_components, max_n_components, h_n_components), record5)
plt.xlabel('PCA特征选取数目')
plt.ylabel('RandomForest-10折交叉验证平均准确率')
ax6 = plt.subplot(2, 3, 6)
plt.plot(range(min_n_components, max_n_components, h_n_components), record6)
plt.xlabel('PCA特征选取数目')
plt.ylabel('BPNN-10折交叉验证平均准确率')
plt.show()
print('over')