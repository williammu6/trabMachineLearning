import pandas as pd
import sys, pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier

train = False if len(sys.argv) < 2 else True

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def converte_categorias(df):
	pd.options.mode.chained_assignment = None  # default='warn'
	
	df.cnpj = pd.Categorical(df.cnpj)
	df['cnpj'] = df.cnpj.cat.codes
	
	df.cfop = pd.Categorical(df.cfop)
	df['cfop'] = df.cfop.cat.codes

	df.ncm = pd.Categorical(df.ncm)
	df['ncm'] = df.ncm.cat.codes
	df.natureza_frete = pd.Categorical(df.natureza_frete)
	df['natureza_frete'] = df.natureza_frete.cat.codes
	return df

folds = 3

sd = SGDClassifier()
sd_params = {}

rl = LogisticRegression()
rl_params = {}

knn = KNeighborsClassifier()
# knn_params = {
# 	"n_neighbors": [2, 3, 5, 10],
# 	"weights": ["uniform", "distance"],
# 	"p": [1, 2]
# }
knn_params = {
	"n_neighbors": [2],
	"weights": ["distance"],
	"p": [2]
}

pcptron = MLPClassifier()
pcptron_params = {
	"hidden_layer_sizes": [1,2],    
	"max_iter": [50,100,200]    
}

rf = RandomForestClassifier()
rf_params = {
	"n_estimators": [5,10],
	"criterion": ["entropy"],
	"max_features": ["auto","sqrt"]
}

dt = DecisionTreeClassifier()
dt_params = {}

classifiers = [sd]
grids = [sd_params]

grid_params = zip(classifiers, grids)

filename = 'model_1.sav'
if train:
	trainfile = './dataset_c1.csv' if len(sys.argv) < 2 else sys.argv[1]
	print('trainfile: ' + trainfile)
	dataset = pd.read_csv(trainfile, delimiter=";")

	trainset = dataset[:int(len(dataset)*0.7)]
	testset = dataset[int(len(dataset)*0.7):]

	X_train = trainset.loc[:, trainset.columns != "y"]
	y_train = trainset.loc[:, trainset.columns == "y"]
	y_train = y_train.values.ravel()


	X_test = testset.loc[:, testset.columns != "y"]
	y_test = testset.loc[:, testset.columns == "y"]
	y_test = y_test.values.ravel()

	X_train = X_train.apply(LabelEncoder().fit_transform)
	X_test = X_test.apply(LabelEncoder().fit_transform)

	# X_train = converte_categorias(X_train)
	# X_test = converte_categorias(X_test)

	for _, (classifier, params) in enumerate(grid_params):

		print("Buscando para algoritmo: {0}\n".format(classifier.__class__))
		
		
		clf = GridSearchCV(estimator=classifier,
								   param_grid=params,
								   cv=folds,
								   n_jobs=-1, 
								   scoring='accuracy') 
					
		clf.fit(X_train, y_train.ravel())

		print("Melhor seleção de hyperparâmetros:\n")
		print(clf.best_params_)
		print("\nScores (% de acertos) nos folds de validação:\n")
		means = clf.cv_results_['mean_test_score'] 
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("{:.3f} (+/-{:.3f}) for {}".format(mean, std * 2, params))
		print("\nResultado detalhado para o melhor modelo:\n")
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true, y_pred))
		pickle.dump(clf, open(filename, 'wb'))
else:
	loaded_model = pickle.load(open(filename, 'rb'))

	evaluate = pd.read_csv('./dataset_c4.csv', delimiter=";")
	
	X_eval = evaluate.loc[:, evaluate.columns != "y"]
	y_eval = evaluate.loc[:, evaluate.columns == "y"]
	y_eval = y_eval.values.ravel()

	X_eval = X_eval.apply(LabelEncoder().fit_transform)
	# X_eval = converte_categorias(X_eval)
	loaded_model.predict(X_eval)
	y_true, y_pred = y_eval, loaded_model.predict(X_eval)
	
	print(classification_report(y_true, y_pred))




