# Code source: João Paulo Dias de Almeida
# Code adapted from: Gaël Varoquaux
# Code original source: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
# License: GNU GPLv3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()

# Pegamos apenas duas caracteristicas de cada amostra
amostras = iris.data[:, :2]

label = iris.target

#Eixos do grafo
x_min, x_max = amostras[:, 0].min() - .5, amostras[:, 0].max() + .5
y_min, y_max = amostras[:, 1].min() - .5, amostras[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))

# Exibe no grafico pontos que correspondem as caracteristicas das amostras
plt.scatter(amostras[:, 0], amostras[:, 1], c=label, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Para observar melhor os dados, foi feita uma transformacao nos dados para
# aumentar as dimensoes com que eles sao representados (2D -> 3D)
# Essa transformacao se chama PCA. Os tres primeiros componentes sao exibidos no grafico
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=label,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Três primeiros componentes PCA")
ax.set_xlabel("1º autovetor")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2º autovetor")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3º autovetor")
ax.w_zaxis.set_ticklabels([])

plt.show()