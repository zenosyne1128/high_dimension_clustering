import numpy as np
import sys
import time
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import metrics
chs = metrics.calinski_harabasz_score

from gen_union_of_subspaces import gen_union_of_subspaces
from metrics.cluster.accuracy import clustering_accuracy
from sklearn import cluster
from sklearn.cluster import SpectralClustering
chs = metrics.calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score as ari

# =================================================
# 定义带噪声的人工数据集make_blobs
# =================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs




n_averages = 50  # how often to repeat classification
n_features_max = 450   # maximum number of features
step = 10  # step size for the calculation


def generate_data(n_samples, features):
    """噪声和数据比例0.4:0.6；中心固定的两分类问题
       返回 数据 和 标签
    """
    new_features = int(0.6 * features)
    a = pow(new_features,-1/4)  #临界值-1/4
    c = np.ones((1,new_features)).reshape(new_features)
    X, y_true = make_blobs(n_samples=n_samples, n_features=new_features, centers=[list(-a*c),list(a*c)])

    if features > 9:
        X = np.hstack([X, np.random.randn(n_samples, int(0.4 * features))])
    return X, y_true

# ================================================================
# 测试特征从10-1000变化时聚类的准确率变化
# =================================================================

tsne_ari, pca_ari, km_ari, sc_ari = [], [], [], []
n_features_range = range(50, n_features_max + 1, step)
for n_features in n_features_range:
    samples = 300
    X, y = generate_data(samples, n_features)
    tsne_proj = TSNE(random_state=1).fit_transform(X)
    tsne_pred = KMeans(n_clusters=2, random_state=9).fit_predict(tsne_proj)
    pca = PCA(n_components=int(n_features/2)).fit_transform(X)
    pca_pred = KMeans(n_clusters=2, n_init=1).fit_predict(pca)
    km_pred = KMeans(init='random', n_clusters=2, n_init=10, algorithm='full').fit_predict(X)
    sc_pred = SpectralClustering(n_clusters=2,affinity='nearest_neighbors',n_neighbors=20).fit_predict(X)

    t_a = ari(y, tsne_pred)
    p_a = ari(y, pca_pred)
    k_a = ari(y, km_pred)
    s_a = ari(y, sc_pred)

    tsne_ari.append(t_a)
    pca_ari.append(p_a)
    km_ari.append(k_a)
    sc_ari.append(s_a)



plt.plot(n_features_range, tsne_ari, linewidth=2,
         label="ARI of t-SNE")
plt.plot(n_features_range, pca_ari, linewidth=2,
         label="ARI of PCA")
plt.plot(n_features_range, km_ari, linewidth=2,
         label="ARI of k-means")
plt.plot(n_features_range, sc_ari, linewidth=2,
         label="ARI of SC")

plt.xlabel('n_features ')
plt.ylabel('Clustering accuracy')

plt.legend( prop={'size': 12})
plt.suptitle('Clustering algrithms on gaussian blobs')
plt.savefig('syn2.png')

# =================================================================
# 450维情况下各具体算法表现
# =================================================================
X, y = generate_data(300, 450)

from scipy import io
from numpy import  mat
X_mat = mat(X)
y_mat = mat(y)
io.savemat("synthetic.mat",{"data": X_mat, "target": y_mat})

ssc = io.loadmat('ssc.mat')
ssc_1 = ssc['ssc']
ssc_pred = ssc_1.reshape(300)
ssc_t = 0
syn = io.loadmat('synthetic.mat')
X1 = syn['data']
y1 = syn['target'].reshape(300)

t1 = time.time()
tsne_proj = TSNE(random_state=1,perplexity= 10).fit_transform(X)
tsne_pred = KMeans(n_clusters=2, random_state=9).fit_predict(tsne_proj)
t2 = time.time()
pca = PCA(n_components=int(450/2)).fit_transform(X)
pca_pred = KMeans(n_clusters=2, n_init=1).fit_predict(pca)
t3 = time.time()
km_pred = KMeans(init='random', n_clusters=2, n_init=10, algorithm='full').fit_predict(X)
t4 = time.time()
sc_pred = SpectralClustering(n_clusters=2,affinity='nearest_neighbors',n_neighbors=20).fit_predict(X)
t5 = time.time()

tsne_ari = ari(y, tsne_pred)
pca_ari = ari(y, pca_pred)
km_ari = ari(y, km_pred)
sc_ari = ari(y, sc_pred)
ssc_ari = ari(y1, ssc_pred)

km_chs = chs(X, km_pred)
tsne_chs = chs(X, tsne_pred)
pca_chs = chs(X, pca_pred)
sc_chs = chs(X, sc_pred)
ssc_chs = chs(X1, ssc_pred)

print('PCA: CHS: {}. ARI: {}. 运行时间: {}'.format(pca_chs, pca_ari, t3-t2))
print('t-SNE: CHS: {}. ARI: {}. 运行时间: {}'.format(tsne_chs, tsne_ari, t2-t1))
print('k-means: CHS: {}. ARI: {}. 运行时间: {}'.format(km_chs, km_ari, t4-t3))
print('谱聚类: CHS: {}. ARI: {}. 运行时间: {}'.format(sc_chs, sc_ari, t5-t4))
print('SSC-ADMM: CHS: {}. ARI: {}. 运行时间: {}'.format(ssc_chs, ssc_ari, ssc_t))

