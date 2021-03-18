from itertools import cycle, islice
from functools import partial

import matplotlib.pyplot as plt
import numpy
import seaborn
from sklearn import preprocessing

from spectral import utils, affinity, clustering
import numpy as np
from  sklearn import  manifold
seaborn.set()
onehot = preprocessing.OneHotEncoder(categories='auto')

methods = [
    (affinity.compute_affinity, 'Basic Affinity'),
    (affinity.com_aff_local_scaling, 'Affinity Local Scaling'),
    # (affinity.automatic_prunning, 'Auto-pruning + LS'),
    # (partial(affinity.automatic_prunning, affinity=affinity.compute_affinity), 'Auto-pruning'),
]
filepath="./train_data/train_old.csv"
data_sets = ['drivingstatus']

num_classes = {
    'drivingstatus': 3
}

#读取文件
def readCSV(filePath):
    try:
        tmp=np.loadtxt(filePath,dtype=np.str,delimiter=",",skiprows=1,encoding='utf-8')
        dataArray=tmp[0:,0:6].astype(np.float)
        label=tmp[0:,6].astype(np.float)
    except OSError:
        print("cant read the csvfile  ")
    else:
        return dataArray
#绘制图片
def drawpic(input,label,colors,plt):
    tsne=manifold.TSNE(n_components=3,init='pca',random_state=501)
    input = tsne.fit_transform(input)
    # ax1 = plt.axes(projection='3d')
    plt.scatter3D(input[:,0],input[:,1],input[:,2],color=colors[label], s=6, alpha=0.6)
    # plt.show()

    print("have finished the pic")

H_plot, W_plot = len(data_sets), len(methods)
fig1=plt.figure(figsize=(8 ,8,))
plt.subplots_adjust(left=0.1, right=0.98, bottom=0.05, top=0.98, wspace=0.2,
                    hspace=0.3)
X = readCSV(filepath)
for i in range(W_plot):
    for j in range(H_plot):
        # X = utils.load_dot_mat('data/DB.mat', 'DB/' + data_sets[j])
        N = X.shape[0]
        K = num_classes.get(data_sets[j], 3)
        affinity, name = methods[i]
        A = affinity(X) #构建相似矩阵
        print("SC on dataset %r with %d classes with method %r" % (data_sets[j], K, name))
        Y = clustering.spectral_clustering(A, K)
        y_pred_label = onehot.fit_transform(Y.reshape(-1, 1)).toarray()
        colors = numpy.array(list(islice(cycle(seaborn.color_palette()), int(max(Y) + 1))))
        t1 = fig1.add_subplot(1, 3, j + 1, projection='3d')
        # ax1 = plt.axes(projection='3d')
        drawpic(X,Y,colors,t1)
        # plt.scatter(X[:, 0], X[:, 1], color=colors[Y], s=6, alpha=0.6)
        if j == 0:
            plt.title(name)

plt.savefig('img/uncentered-auto.png')
