# Code source: Joao Paulo Dias de Almeida
# License: GNU GPLv3

import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Faz o download do dataset Iris
iris = datasets.load_iris()

# Pegamos apenas duas caracteristicas de cada amostra
amostras = iris.data[:, :2]

label = iris.target

Amostras_treino, Amostras_teste, Label_treino, Label_teste = train_test_split(
    amostras, label, test_size=0.25, random_state=10)

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(Amostras_treino, Label_treino)

pred = knn.predict(Amostras_teste)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]),decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


c2 = confusion_matrix(Label_teste, pred)

class_names = ['Versicolor', 'Setosa', 'Virginica']

plt.figure()
plot_confusion_matrix(c2, classes=class_names, normalize=False, title='Confusion matrix')