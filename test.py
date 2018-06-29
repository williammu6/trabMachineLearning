from config import *

loaded_model = pickle.load(open(filename, 'rb'))

evaluate = pd.read_csv('./data/' + sys.argv[1], delimiter=";")
evaluate = evaluate.loc[:, evaluate.columns != 'cnpj']
evaluate = evaluate.drop(evaluate[evaluate.y == 'monofasico aliquotas diferenciadas'].index)

X_eval = evaluate.loc[:, evaluate.columns != "y"]
y_eval = evaluate.loc[:, evaluate.columns == "y"]
y_eval = y_eval.values.ravel()

class_names = evaluate.y.unique()

X_eval = X_eval.apply(LabelEncoder().fit_transform)
X_eval = scaler.fit_transform(X_eval)

loaded_model.predict(X_eval)
y_true, y_pred = y_eval, loaded_model.predict(X_eval)

print(classification_report(y_true, y_pred))

plt.figure()

matrix = confusion_matrix(y_true, y_pred)

plot_confusion_matrix(matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()

