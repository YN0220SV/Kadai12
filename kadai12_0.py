

# ライブラリをインポート
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import seaborn as sns

plt.rcParams['font.family'] = 'IPAexMincho'
# weatherデータ(csv形式)の読み込み
weather_df = pd.read_csv('/home/yn/デスクトップ/Kadai12/data3.csv')

weather_data = weather_df[[ "最高気温", "日照時間",  "平均湿度"]]
weather_df2 = weather_df[["最高気温", "日照時間",  "平均湿度", '天気概況改']]
weather_target = weather_df['天気概況改']
data=list(set(weather_target))
data = ["最高気温", "日照時間",  "平均湿度"]
xlist=[]
ylist=[]


# クラスタ数を決定するために距離の総計を確認
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='random', max_iter=30, random_state=1)
    km.fit(weather_data)
    distortions.append(km.inertia_)


plt.figure(figsize=(10, 4))
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# クラスタ数を決めて、kmeans法でクラスタリング
_n_cluster = 3
kmeans = KMeans(n_clusters=_n_cluster, max_iter=30,init="random", random_state=1)
cluster = kmeans.fit_predict(weather_data)

weather_result=weather_df2.copy()
weather_result["cluster"]=["cluster"+str(x) for x in cluster]
sns.pairplot(weather_result,hue="cluster",height=1.7)

plt.show()
sns.pairplot(weather_df2,hue="天気概況改",height=1.7)


plt.show()

# 各クラスタの天気概況を確認
for i in range(_n_cluster):
    _weather_list = []
    for _cluster, _weather in zip(cluster, weather_target):
        if i == _cluster:
            _weather_list.append(_weather)
    print('cluster', i, _weather_list)





# 各クラスターの重心を棒グラフで表示
# (データサイエンスのための統計学入門 p316 より)
centers = pd.DataFrame(kmeans.cluster_centers_, columns=data)
f, axes = plt.subplots(_n_cluster, 1, sharex=True, figsize=(10, 10))
for i, ax in enumerate(axes):
    center = centers.loc[i, :]
    maxPC = 1.1 * np.max(np.max(np.abs(center)))
    center.plot.bar(ax=ax)
    ax.axhline(color='#cccccc')
    ax.set_ylabel(f'Cluster {i+1}')
    ax.set_ylim(-maxPC, maxPC)
plt.show()

# 混合カウスモデルでクラスタリングを実行
_n_cluster = 3
clf = GaussianMixture(n_components=_n_cluster)
cluster = clf.fit_predict(weather_data)
weather_result = weather_df2.copy()
weather_result["cluster"] = ["cluster"+str(x) for x in cluster]
sns.pairplot(weather_result, hue="cluster", height=1.7)


# 各クラスタの天気概況を確認
for i in range(_n_cluster):
    _weather_list = []
    for _cluster, _weather in zip(cluster, weather_target):
        if i == _cluster:
            _weather_list.append(_weather)
    print('cluster', i, _weather_list)

# 各クラスターの重心を棒グラフで表示
# (データサイエンスのための統計学入門 p316 より)
centers = pd.DataFrame(clf.means_, columns=data)
f, axes = plt.subplots(_n_cluster, 1, sharex=True, figsize=(10, 10))
for i, ax in enumerate(axes):
    center = centers.loc[i, :]
    maxPC = 1.1 * np.max(np.max(np.abs(center)))
    center.plot.bar(ax=ax)
    ax.axhline(color='#cccccc')
    ax.set_ylabel(f'Cluster {i+1}')
    ax.set_ylim(-maxPC, maxPC)

plt.show()