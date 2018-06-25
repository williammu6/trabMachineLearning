import pandas as pd
import sys, pickle
from sklearn.preprocessing import *
from sklearn.linear_model import SGDClassifier
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
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

count_vect = CountVectorizer()

folds = 3
filename = 'model_1.sav'

mlp = MLPClassifier()
mlp_params = {
	"activation": ["identity", "logistic", "tanh", "relu"]
}

kmeans = KMeans()
kmeans_params = {
	"n_clusters": [2, 5],
	"random_state": [0],
	"max_iter": [100, 200, 300]
}

scaler = StandardScaler()

sd = SGDClassifier()
sd_params = {
	"loss": ["hinge"], 
	"penalty": ["l2"],
	"alpha": [1e-3], 
	"random_state": [42],
	"max_iter": [5], 
	"tol": [None],
	"loss": ["log"]
}

svr = SVR()
svr_params = {
	"kernel": ["rbf", "linear", "poly"],	
	"C": [1e3],
	"gamma": [0.1],
	"degree": [2]
}


rl = LogisticRegression()
rl_params = {}

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

classifiers = [mlp]
grids = [mlp_params]

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
