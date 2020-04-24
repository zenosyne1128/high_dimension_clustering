import numpy as np
import sys
import time
from sklearn.preprocessing import scale

from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from gen_union_of_subspaces import gen_union_of_subspaces
from metrics.cluster.accuracy import clustering_accuracy
from sklearn import cluster

# =================================================
# 载入独立子空间数据集 并三维可视化
# =================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gen_union_of_subspaces import gen_union_of_subspaces

ambient_dim = 9
subspace_dim = 6
num_subspaces = 5
num_points_per_subspace = 50

X, y_true = gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, 0.00)
y_true = y_true.reshape(250)
X = scale(X)
fig = plt.figure()
ax1 = Axes3D(fig)
ax1.scatter3D(X[:,1],X[:,2],X[:,3], c=y_true)  #绘制散点图
ax1.legend
# plt.savefig("unsp1.png",bbox_inches='tight')
plt.show()

# ================================================================================
# 定义二维带标签可视化函数
# ================================================================================
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def scatter(x, colors, title = None, label = True):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 5))

    # We create a scatter plot.
    f = plt.figure(figsize=(8,8))

    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    if label == True:
        for i in range(5):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    if title != None:
        tit = plt.title(title)
    else: tit = None
    return f, ax, sc, txts, tit


# ========================================================================
# t-SNE + k-means 聚类效果
# =========================================================================
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
RS = 1
t_3 = time.time()
tsne_proj = TSNE(random_state=RS, perplexity= 15).fit_transform(X)
print(tsne_proj.shape)
t_4 = time.time()
ax1 = scatter(tsne_proj, y_true, "t-SNE visualization of raw data")
# plt.savefig('unsp2.png',bbox_inches='tight')
plt.show()


calinski = []
# k-means 测试选取最佳聚类数

for i in range(1,10):
        y_pred = KMeans(n_clusters=i, random_state=9).fit_predict(tsne_proj)
        score = ari(y_true, y_pred)
        calinski.append(score)

fig = plt.figure()
ax =fig.add_subplot(1,1,1)
ax.plot(range(1, 10), calinski)
ax.set_xlabel('n_cluster')
ax.set_ylabel('CHS')
plt.show()   #观察选取最佳聚类数
# plt.savefig('wtrend.png', bbox_inches = 'tight')

t_1 = time.time()
tsne_pred = KMeans(n_clusters=5, random_state=9).fit_predict(tsne_proj) #k-means 结果
t_2 = time.time()
tsne_t = t_2-t_1 +t_4-t_3   #计算运行时间

ax2 = scatter(tsne_proj, tsne_pred, "clustering result of t-SNE") #可视化聚类结果
plt.show()
# plt.savefig('sne_mn.png')
# =================================================================================
# PCA+ kmeans 和直接 k-means 聚类效果与可视化
# =================================================================================
from sklearn.decomposition import PCA
from sklearn import metrics
chs = metrics.calinski_harabasz_score

pca = PCA(n_components=6)
pca.fit(X)
print(pca.explained_variance_ratio_)
pca = PCA(n_components=0.95)
pca.fit(X)
print(pca.n_components_)

ratiorange = np.arange(0.5, 1, 0.01)
ncomp = []
for r in ratiorange:
    pca = PCA(n_components=r)
    pca.fit(X)
    k = pca.n_components_
    ncluster = ncomp.append(k)

fig = plt.figure()
ax =fig.add_subplot(1,1,1)
ax.plot(ncomp, ratiorange)
ax.set_xlabel('n_components')
ax.set_ylabel('variance ratio')
plt.show()   #观察选取最佳维数
# plt.savefig('upca.png', bbox_inches='tight')


pca = PCA(n_components=7).fit(X)
t1 = time.time()
pca_pred = KMeans(init=pca.components_, n_clusters=7, n_init=1).fit(X)
pca_pred = pca_pred.labels_
t2 = time.time()
km_pred = KMeans(init='random', n_clusters=5, n_init=10).fit(X)
km_pred = km_pred.labels_
#这里调用传统KMeans算法
t3 = time.time()
pca_t = t2-t1
km_t = t3-t2

# pca_proj2 = PCA(n_components=2).transform(X)
# pca_pred2 = KMeans(n_clusters=10, random_state=9).fit_predict(pca_proj2)

reduced_data = PCA(n_components=5).fit_transform(X)
kmeans = KMeans(init='k-means++', n_clusters=5, n_init=5)
kmeans.fit(reduced_data)
pca_pred2 = kmeans.labels_
# scatter(reduced_data, pca_pred2, "clustering result of PCA")
# plt.show()
# plt.savefig('pca_mn.png')

# ================================================================================
# 谱聚类
# ================================================================================

from sklearn import metrics
from sklearn.cluster import SpectralClustering
chs = metrics.calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score as ari


neig = np.arange(5, 50, 5)
ychs = []

t1 = time.time()
sc_pred = SpectralClustering(n_clusters=7,affinity='nearest_neighbors',n_neighbors=6).fit_predict(X)
t2 = time.time()
sc_t = t2 - t1
# ===================================================================
# 在MATLAB中使用SSC-ADMM求解并将结果载入pycharm
# =====================================================================
from scipy import io
from numpy import mat
X_mat = mat(X)
y_mat = mat(y_true)
io.savemat("unsp.mat",{"data": X_mat, "target": y_mat})

ssc = io.loadmat('SSC2.mat')
ssc_1 = ssc['ssc']
ssc_pred = ssc_1.reshape(250)
ssc_t = 0

# =================================================================================
# 最优聚类结果评判与对比
# ==================================================================================
from sklearn.metrics import adjusted_rand_score as ari
from sklearn import metrics

chs = metrics.calinski_harabasz_score


# y_pred =(
#     ('PCA', pca_pred)
#     ('KMeans', km_pred)
#     #('Spectral Clustering',sc_pred)
#     #('t-SNE', tsne_pred)
# )
# for name, label_pred in y_pred:
#     ARI = adjusted_rand_score(y_true,label_pred)
#     CHI = metrics.calinski_harabasz_score(X,label_pred)
#
#     print('算法: {}. ARI: {}. CHI: {}'.format(name, ARI, CHI))
km_chs = chs(X, km_pred)
tsne_chs = chs(X, tsne_pred)
pca_chs = chs(X, pca_pred)
sc_chs = chs(X, sc_pred)
ssc_chs = chs(X, ssc_pred)

pca_ari = ari(y_true, pca_pred)
km_ari = ari(y_true, km_pred)
tsne_ari = ari(y_true, tsne_pred)
sc_ari = ari(y_true, sc_pred)
ssc_ari = ari(y_true, ssc_pred)

print('PCA: CHS: {}. ARI: {}. 运行时间: {}'.format(pca_chs, pca_ari, pca_t))
print('t-SNE: CHS: {}. ARI: {}. 运行时间: {}'.format(tsne_chs, tsne_ari, tsne_t))
print('k-means: CHS: {}. ARI: {}. 运行时间: {}'.format(km_chs, km_ari, km_t))
print('谱聚类: CHS: {}. ARI: {}. 运行时间: {}'.format(sc_chs, sc_ari, sc_t))
print('SSC-ADMM: CHS: {}. ARI: {}. 运行时间: {}'.format(ssc_chs, ssc_ari, ssc_t))

