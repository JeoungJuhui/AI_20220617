
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from matplotlib import pyplot as plt

from pandas import read_csv

df = read_csv('vgsales.csv')
data = df.values
print(data[:1])

plt.scatter(data[:,2], data[:,10], c="white", marker='o', edgecolors='black',s=50)
#플랫폼에 따른 글로벌 판매량
plt.grid()
plt.tight_layout()
plt.show()


df2=read_csv('Test.csv')
data2=df2.values




plt.scatter(data2[:,1], data2[:,2], c="white", marker='o', edgecolors='black',s=50)
plt.grid()
plt.tight_layout()

#국가별판매량 k-means
kmeans=KMeans(n_clusters=3,init="random",n_init=10,max_iter=300,random_state=0)
pred2=kmeans.fit_predict(data2)
print(pred2)


'''
x, y=make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,shuffle=0)
#임의의 데이터 set을 만들어낸다.

plt.scatter(x[:,0], x[:,1], c="white", marker='o', edgecolors='black',s=50)
plt.grid()
plt.tight_layout()
plt.show()
'''



'''
kmeans=KMeans(n_clusters=3,init="random",n_init=10,max_iter=300,random_state=0)
pred=kmeans.fit_predict(x)
'''