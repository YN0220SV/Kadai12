# ライブラリをインポート
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import openpyxl
import pandas as pd
import datetime
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


weather_df = pd.read_csv('/home/yn/デスクトップ/Kadai12/data3.csv')


year = 2020
month = 1
data = ['平均気温', '最高気温', '最低気温', '降水量', '日照時間', '降雪量', '平均風速', '平均蒸気圧', '平均湿度', '平均現地気圧'] 
target = '天気概況'
weather_data = weather_df[data]
weather_target = weather_df['天気概況改']

print(weather_data)
print(weather_target)


# データを標準化
stdsc = StandardScaler()
weather_data = stdsc.fit_transform(weather_data)


#主成分分析を実行
pca = PCA()
pca.fit(weather_data)

# データを主成分空間に写像
pca_cor = pca.transform(weather_data)

# 第一主成分と第二主成分で各日を天気概況付きでプロット
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca_cor[:, 0], pca_cor[:, 1], weather_target):
    #plt.text(x+0.05, y, name)
    pass


draw=pd.DataFrame()
draw["pc0"] = pca_cor[:, 0]
draw["pc1"] = pca_cor[:, 1]
draw["tgt"]=weather_target
sns.scatterplot(x="pc0",y="pc1",hue="tgt",data=draw)
#plt.scatter(pca_cor[:, 0], pca_cor[:, 1])


plt.grid()
plt.xlabel("PC0")
plt.ylabel("PC1")
plt.show()

# PCA の固有ベクトル
pd.DataFrame(pca.components_, columns=data, 
             index=["PC{}".format(x) for x in range(len(data))])


# 固有ベクトルを棒グラフで表示
centers = pd.DataFrame(pca.components_, columns=data)
f, axes = plt.subplots(len(data), 1, sharex=True, figsize=(10, 10))
for i, ax in enumerate(axes):
    center = centers.loc[i, :]
    maxPC = 1.1 * np.max(np.max(np.abs(center)))
    center.plot.bar(ax=ax)
    ax.axhline(color='#cccccc')
    ax.set_ylabel(f'PC{i}')
    ax.set_ylim(-maxPC, maxPC)


# 第一主成分と第二主成分における観測変数をプロット
plt.figure(figsize=(6, 6))
origin = (0, 0)
for x, y, name in zip(pca.components_[0], pca.components_[1], data):
    plt.text(x, y, name)
    plt.annotate(text='', xy=(x, y), xytext=origin,arrowprops=dict(shrink=0, width=1, headwidth=8, headlength=10,connectionstyle='arc3', facecolor='lightgray', edgecolor='lightgray'))
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC0")
plt.ylabel("PC1")
plt.show()


# 寄与率
for i in range(len(data)):
    print(f'PC{i} {pca.explained_variance_ratio_[i]:.06f}')


# 累積寄与率
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.grid()
plt.show()
