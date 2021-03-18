import numpy as np
import matplotlib.pyplot as plt
from  sklearn import  manifold
from itertools import cycle, islice
import seaborn

def drawpic(input,label,colors,plt):
    tsne=manifold.TSNE(n_components=3,init='pca',random_state=501)
    input = tsne.fit_transform(input)
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(input[:,0],input[:,1],input[:,2],color=colors[label], s=6, alpha=0.6)
    # plt.show()

    print("have finished the pic")


colors = np.array(list(islice(cycle(seaborn.color_palette()), 3)))

fg1=plt.figure(figsize=(8,8))
plt.subplots_adjust(left=0.1, right=0.98, bottom=0.05, top=0.98, wspace=0.2,hspace=0.3)
a1=np.random.randint(0,10,size=[200,3])
b1=np.random.randint(0,2,size=[200])
for i in range(3):
    # plt1 = plt.subplot(1,3, i+1)
    t1=fg1.add_subplot(1,3,i+1, projection='3d')
    t1.scatter3D(a1[:,0],a1[:,1],a1[:,2],color=colors[b1], s=6, alpha=0.6)
    plt.title(i)
plt.show()
print("aa")
