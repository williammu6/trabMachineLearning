from config import *

if len(sys.argv) < 2:
	print("Usage: python3 test.py <test_file>")
	exit()

loaded_model = pickle.load(open(filename, 'rb'))

evaluate = pd.read_csv(sys.argv[1], delimiter=";")

X_eval = evaluate.loc[:, evaluate.columns != "y"]
y_eval = evaluate.loc[:, evaluate.columns == "y"]
y_eval = y_eval.values.ravel()

X_eval = X_eval.apply(LabelEncoder().fit_transform)
loaded_model.predict(X_eval)
y_true, y_pred = y_eval, loaded_model.predict(X_eval)

print(classification_report(y_true, y_pred))
