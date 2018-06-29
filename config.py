import pandas as pd
import sys, pickle, itertools
import numpy as np
import scikitplot as skplt
from sklearn.preprocessing import *
# from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# count_vect = CountVectorizer()
scaler = StandardScaler()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def select_k_best_features(method, train_features, test_features, targets, k_best=10):
	selector = SelectKBest(method, k=k_best).fit(train_features, targets)
	indices = np.where(selector.get_support() == True)
	new_train_features = selector.transform(train_features)
	new_test_features = selector.transform(test_features)
	return new_train_features, new_test_features, indices


folds = 3

filename = './model/goodmodel.sav'

mlp = MLPClassifier()
mlp_params = {
	# "activation": ["logistic", "tanh", "relu"],
	# "learning_rate": ['adaptive', 'constant', 'invscaling']
	"activation": ["logistic"],
	"hidden_layer_sizes": [100], # 2, 3, 
}

knn = KNeighborsClassifier()
knn_params1 = {
	"n_neighbors": [2, 3, 5, 10],
	"weights": ["uniform", "distance"],
	"p": [1, 2]
}
knn_params2 = {
	"n_neighbors": [2],
	"weights": ["distance"],
	"p": [2]
}


classifiers = [mlp]
grids = [mlp_params]