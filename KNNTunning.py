# Code source: Joao Paulo Dias de Almeida
# License: GNU GPLv3

from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import gc

tuned_parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}]

# Vou escolher o modelo com maior precisao
# Voce pode mudar isso. Como e um vetor, voce pode testar varias caracteristicas ao mesmo tempo
scores = ['precision']

iris = datasets.load_iris()

X = iris.data[:, :2]

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10)

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=10,
                       scoring='%s_macro' % score)
    gc.collect()
    clf.fit(X_train, y_train)

    #Aqui ele vai mostrar o melhor valor de k encontrado, e mostrar a precision, recall e f-score para cada label. No Iris dataset sao 3
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()