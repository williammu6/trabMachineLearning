from config import *

grid_params = zip(classifiers, grids)

if len(sys.argv) < 2:
	print("Usage: python3 train.py <train_file>")
	exit()

dataset = pd.read_csv(sys.argv[1], delimiter=";")

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

# X_train = tf_transformer.transform(X_train)
# X_test = tf_transformer.transform(X_test)

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

	# for doc, real, pred in zip(	, y_true, y_pred):
	# 	print("%s => %s => %s" % (doc,  real, pred))

	# pickle.dump(clf, open(filename, 'wb'))



