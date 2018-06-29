from config import *

if len(sys.argv) < 2:
	print("Usage: python3 train.py <train_file>")
	exit()

dataset = pd.read_csv('./model/' + sys.argv[1], delimiter=";")
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset = dataset.drop(dataset[dataset.y == 'monofasico aliquotas diferenciadas'].index)

class_names = dataset.y.unique()

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

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Função para testar combinações de features e seus resultados

# for n in range(1,4):
	# print("Feature: " + str(n))
	# selected_feature, selected_test_features, best = select_k_best_features(f_classif, X_train, X_test, y_train, k_best=n)
	# best_feature_names = X_train.columns[best]
	# grid_params = zip(classifiers, grids)

grid_params = zip(classifiers, grids)

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
	# skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
	print(classification_report(y_true, y_pred))
	# print(best_feature_names)
	# for doc, real, pred in zip(	, y_true, y_pred):
	# 	print("%s => %s => %s" % (doc,  real, pred))

	plt.figure()
	matrix = confusion_matrix(y_true, y_pred)
	pickle.dump(clf, open(filename, 'wb'))

# skplt.metrics.plot_precision_recall_curve(X_test, probas)

plot_confusion_matrix(matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()
