# imports
from numpy.lib.function_base import select
import pandas as pd
import numpy as np
from pandas.core.base import SelectionMixin
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

#sklearn.preprocessing.normalize ??? je sais pas si il faut

# verbose
VERBOSE = True

"""
Comments on the project :
	- As of python 3.5 (I think), you might see datatypes given explicitely (x : int = 3). This is only used for code clarity as python will not care about the explicit type given (x : int = 3.2 is perfectly valid and type(x) will return 'float').

? Questions to ask :
	?

Documentation to read :
	DOC

"""

"""
Functions from the project brief
"""
def score_f1(y_true, y_pred, th):
	return metrics.f1_score(y_true > th, y_pred > th)

def score_regression(y_true , y_pred):
	scores = [ score_f1(y_true, y_pred, th=th) for th in [500, 1400, 5000, 10000] ]
	return np.mean(scores)

"""
Functions not depending on the class instance
"""
def normalize_data_as_panda(data : pd.DataFrame, scaler = StandardScaler()) -> pd.DataFrame:
	"""
	Function to normalise a panda DataFrame.
	@param:	
		data: a panda [DataFrame] to normalize
		scaler: the scaler to use
	@returns: a normalized panda [DataFrame]
	"""
	normalized_data = scaler.fit_transform(data)
	pd_normalized_data = pd.DataFrame(data=normalized_data, index=data.index, columns=data.columns)
	assert np.all(normalized_data == pd_normalized_data.values)
	return pd_normalized_data

"""
Class used to be able to easily access different values and pass information between functions
"""
class Project:

	def __init__(self):
		# WIP
		if VERBOSE: print("\n--- Preprocessing ---")
		self.read_data()
		# ! Bri et Val font un normalize sur tout, preprocessing sur tout (pas oublier X2) et puis split et puis fit
		# TODO preprocess jours en sin cos
		self.split_data()
		self.normalize_data()

		if VERBOSE: print("\n--- Feature Selection ---")
		print("no feature selection implemented yet")

		if VERBOSE: print("\n--- Splitting data ---")
		# ensuite on peut split nos data
		# self.X_trainScaled, self.X_testScaled = self.normalize_data(self.X_train, self.X_test)

	def read_data(self,
		X1_file : str = "X1.csv",
		Y1_file : str = "Y1.csv") -> None:
		"""
		read the data from the input files
		"""

		# load the input and output files
		self.X1 : pd.DataFrame = pd.read_csv(X1_file)
		self.Y1 : pd.DataFrame = pd.read_csv(Y1_file, header=None, names=['shares '])
		if VERBOSE : print(f"Data has been retrieved from files '{X1_file}' and '{Y1_file}'")

	def split_data(self, 
		test_size : float = 0.20):
		"""
		split the data between training and testing
		"""
		self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(self.X1, self.Y1, test_size=test_size)

		if VERBOSE : print(f"Data has been split between train and test, with {test_size*100}% of the data used as testing data")

	def normalize_data(self):
		"""
		Normalize the data 
		DOC : https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
		"""
		# normalize the full data
		if hasattr(self, 'X1'):	
			scaler = StandardScaler()
			scaler.fit(self.X1)
			self.X1_scaled = scaler.transform(self.X1)
			if VERBOSE : print("X1 has been normalized")
		elif VERBOSE : print("Instance has no attribute 'X1', try running read_data() before normalizing")

		if hasattr(self, 'X_train') and hasattr(self, 'X_test'):
			scaler = StandardScaler()
			scaler.fit(self.X_train)
			self.X_train_scaled = scaler.transform(self.X_train)
			self.X_test_scaled  = scaler.transform(self.X_test)
			if VERBOSE : print("X_train and X_test have been normalized based on X_train")
		elif VERBOSE : print("Instance has no attribute 'X_train' or 'X_test' (probably both)")
	
	def describe_features(self):
		"""
		print a description of the features
		"""
		print(self.X1.describe)

	def get_features_names(self):
		"""
		print the names of the different features
		"""
		print(self.X1.columns)

	def plot_correlation_matrix(self, filename : str = "correlation_mat.svg", normalize : bool = True):
		if normalize:
			X1_normalized = normalize_data_as_panda(self.X1)
			correlation_mat = X1_normalized.join(self.Y1).corr()
		else :
			correlation_mat = self.X1.join(self.Y1).corr()
		plt.subplots(figsize=(25,20))
		sns.heatmap(correlation_mat, annot = False)
		plt.savefig(filename)
		if VERBOSE : print(f"Saved correlation matrix to '{filename}'")

	def remove_correlation_features(self, th=85):
		# WIP I suppose...
		cor = np.abs(np.corrcoef(self.X1_scaled, self.Y1))
		upperCor = np.triu(cor, k=1)	#k=1 to ignore the diagonal

	def predict_with_linear_regression(self, scaled = True):
		"""
		Fit a linear regressor using the training data and make a prediction of the test data
		"""
		linear_regressor = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
		if scaled:
			linear_regressor.fit(self.X_train_scaled, self.Y_train)
			prediction = linear_regressor.predict(self.X_test_scaled)
		else:
			linear_regressor.fit(self.X_train, self.Y_train)
			prediction = linear_regressor.predict(self.X_test)
		return prediction, linear_regressor.coef_

	def predict_with_lasso(self, scaled = True, max_iter : int = 1100, tol : float = 1e-4, warm_start : bool = False):
		"""
		Linear model trained with L1 prior as regularizer (aka the Lasso). 
		DOC : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso:
		"""
		lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, max_iter=max_iter, tol=tol, warm_start=warm_start, selection='random')
		# we can also use selection='cyclic' to loop over features sequentially
		if scaled:
			lasso.fit(self.X_train_scaled, self.Y_train)
			prediction = lasso.predict(self.X_test_scaled)
		else:
			lasso.fit(self.X_train, self.Y_train)
			prediction = lasso.predict(self.X_test)
		return prediction, lasso.coef_
	
	def predict_with_knn(self, scaled = True, n_neighbors : int = 5, weights = 'uniform', algorithm = 'auto', leaf_size : int = 30, p : int = 2):
		"""
		Prediction with KNN
		DOC : https://scikit-learn.org/stable/modules/neighbors.html#neighbors (important for undersanding the choice of algorithm to use)
		@param weights : {'uniform', 'distance'}
		@param algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
		? @Gauthier p pour manhattan_distance ou euclidean_distance ou autre pour minkowski dependant de p
		"""
		# other parameters for KNeighborsRegressor : metric='minkowski', metric_params=None
		knn = KNeighborsRegressor(n_neighbors, weights, algorithm, leaf_size, p, n_jobs=-1)
		if scaled:
			knn.fit(self.X_train_scaled, self.Y_train)
			prediction = knn.predict(self.X_test_scaled)
		else:
			knn.fit(self.X_train, self.Y_train)
			prediction = knn.predict(self.X_test)
		return prediction

	def predict_with_mlp(self, scaled = True):
		"""
		Prediction with MLP
		DOC : https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
		"""
		mlp = MLPRegressor(
			hidden_layer_sizes=(100, ), # 1 layer with 100 neurons
			activation='relu',          # {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
			solver='adam',              # {‘lbfgs’, ‘sgd’, ‘adam’} # sgd lbfgs pour petits sets, adam robuste, sgd donne de meilleurs resultats si le learning rate est bien reglé
			alpha=1e-4,		              # regularization term
			batch_size='auto',
			learning_rate='constant',	  # {‘constant’, ‘invscaling’, ‘adaptive’}
			learning_rate_init=1e-3,
			power_t=0.5,								# only used when solver=’sgd’.
			max_iter=200,
			shuffle=True,		            # only used when solver=’sgd’ or ‘adam’
			random_state=None,
			tol=1e-4,
			verbose=False,
			warm_start=False,
			momentum=0.9,
			nesterovs_momentum=True,
			early_stopping=False,
			validation_fraction=0.1,
			beta_1=0.9,
			beta_2=0.999,
			epsilon=1e-08,
			n_iter_no_change=10,
			max_fun=15000)
		if scaled:
			mlp.fit(self.X_train_scaled, self.Y_train)
			prediction = mlp.predict(self.X_test_scaled)
		else:
			mlp.fit(self.X_train, self.Y_train)
			prediction = mlp.predict(self.X_test)
		return prediction

	def get_grid_search_knn(self, scaled = True):
		"""
		DOC : https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
		"""
		scoring = {
			'NegMSE': 'neg_mean_squared_error', 
			'score_regression': metrics.make_scorer(score_regression, greater_is_better=True)
		}
		grid = {
			'n_neighbors': [3, 5, 7, 9, 13, 15, 20, 30],
			'weights': ['uniform', 'distance']
		}

		# parameter refit='score_regression', refits an estimator on the whole dataset with the parameter setting that has the best cross-validated score_regression score
		gs = model_selection.GridSearchCV(
			KNeighborsRegressor(algorithm='auto', n_jobs=-1), 
			param_grid=grid,
			scoring=scoring, 
			refit='score_regression', 
			return_train_score=True, 
			error_score=0, 
			n_jobs=-1, 
			verbose=3)
		if scaled:
			gs.fit(self.X_train_scaled, self.Y_train)
		else:
			gs.fit(self.X_train, self.Y_train)

		if VERBOSE:
			print("--- Grid search KNN ---")
			print("best parameters:", gs.best_params_)
			print("training score:", gs.best_score_)

		return gs

	def get_grid_search_mlp(self, scaled = True):
		scoring = {
			'NegMSE': 'neg_mean_squared_error', 
			'score_regression': metrics.make_scorer(score_regression, greater_is_better=True)
		}
		grid = {
			'hidden_layer_sizes': [(100,), (140,), (50,50,), (50,)],
			'activation': ['identity', 'logistic', 'tanh', 'relu'],
			'solver': ['adam'],
			'alpha': 10.0 ** -np.arange(1, 7) 
			# DOC : alpha advised by https://scikit-learn.org/stable/modules/neural_networks_supervised.html
			# TODO
		}

		gs = model_selection.GridSearchCV(
			MLPRegressor(),
			param_grid=grid,
			scoring=scoring, 
			refit='score_regression', 
			return_train_score=True, 
			error_score=0, 
			n_jobs=-1, 
			verbose=3)
		if scaled:
			gs.fit(self.X_train_scaled, self.Y_train)
		else:
			gs.fit(self.X_train, self.Y_train)
		
		if VERBOSE:
			print("--- Grid search MLP ---")
			print("best params:", gs.best_params_)
			print("training score:", gs.best_score_)

		return gs





p = Project()

# linear regression not scaled
prediction,_ = p.predict_with_linear_regression(scaled=False)
print("score by LinearRegression testing (not scaled):", score_regression(p.Y_test, prediction))	# 0.48896528584814974

# lasso not scaled
prediction,_ = p.predict_with_lasso(scaled=False)
print("score by Lasso testing (not scaled):", score_regression(p.Y_test, prediction))	# 0.488894300572261

# linear regression scaled
prediction,_ = p.predict_with_linear_regression()
print("score by LinearRegression (scaled):", score_regression(p.Y_test, prediction)) # 0.48243917542285	

# lasso scaled
prediction,_ = p.predict_with_lasso()
print("score by Lasso testing (scaled):", score_regression(p.Y_test, prediction))	# 0.4834171812808842


A = np.corrcoef(p.X_train, p.Y_train, rowvar=False)
B = np.corrcoef(p.X_train_scaled, p.Y_train, rowvar=False)
print(A[1,2], B[1,2])

#/!\ preneur en temps mais beau et instructif sur les features donnant potentiellement les mêmes info.
#proj.plotCorrelationMatrix(filename="correlation_matNotNormalized.svg", normalize=False)
#proj.plotCorrelationMatrix()

#gsMLP = getGridSearchMLP(proj)
#MLPprediction = gsMLP.predict(proj.X_testScaled)
#print(f"score by LinearRegression: {proj.score_regression(proj.Y_test, MLPprediction)}")



"""
Documentation to read for next time :
	DOC : https://scikit-learn.org/stable/modules/grid_search.html#grid-search
	DOC : https://scikit-learn.org/stable/modules/neighbors.html#neighbors -> le bas est potentiellement très important pour la features selection!
	DOC : https://scikit-learn.org/stable/modules/neural_networks_supervised.html -> voir les tips en bas, dit ce qu'il faut mettre dans grid_search + dit qu'on triche en fittant le scaller sur le trainSet et testSet.

? Questions :
	? shuffle ou pas??
	? If the data ordering is not arbitrary (e.g. samples with the same class label are contiguous), shuffling it first may be essential to get a meaningful cross-validation result.
	? However, the opposite may be true if the samples are not independently and identically distributed. For example, if samples correspond to news articles, and are ordered by their time of publication,
	? then shuffling the data will likely lead to a model that is overfit and an inflated validation score: it will be tested on samples that are artificially similar (close in time) to training samples.
"""