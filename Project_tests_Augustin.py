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
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import KernelPCA

#sklearn.preprocessing.normalize ??? je sais pas si il faut

# verbose
VERBOSE = True

"""
Comments on the project :
	- As of python 3.5 (I think), you might see datatypes given explicitely (x : int = 3). This is only used for code clarity as python will not care about the explicit type given (x : int = 3.2 is perfectly valid and type(x) will return 'float').

? Questions to ask :
	?

! Important notes :
	! Bri et Val font un normalize sur tout, preprocessing sur tout (pas oublier X2) et puis split et puis fit

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
		self.normalize_data()

		if VERBOSE: print("\n--- Feature Selection ---")
		self.feature_selection()

		if VERBOSE: print("\n--- Splitting data ---")
		self.split_data()

		# ensuite on peut split nos data
		# self.X_trainScaled, self.X_testScaled = self.normalize_data(self.X_train, self.X_test)

	def feature_selection(self):
		self.days_one_hot_to_sin_cos()

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
			self.X1_scaled = scaler.fit_transform(self.X1)
			if VERBOSE : print("X1 has been normalized")
		elif VERBOSE : print("Instance has no attribute 'X1', try running read_data() before normalizing")
	
	def days_one_hot_to_sin_cos(self):
		# transform one-hot into list : monday = 0, tuesday = 1, ...
		l = np.zeros(len(self.X1['weekday_is_monday']))
		for idx, mon, tue, wed, thu, fri, sat, sun in zip(range(len(l)),
			self.X1['weekday_is_monday'], 
			self.X1['weekday_is_tuesday'],
			self.X1['weekday_is_wednesday'],
			self.X1['weekday_is_thursday'],
			self.X1['weekday_is_friday'],
			self.X1['weekday_is_saturday'],
			self.X1['weekday_is_sunday']):

			if   mon == 1 : l[idx] = 0; continue
			elif tue == 1 : l[idx] = 1; continue
			elif wed == 1 : l[idx] = 2; continue
			elif thu == 1 : l[idx] = 3; continue
			elif fri == 1 : l[idx] = 4; continue
			elif sat == 1 : l[idx] = 5; continue
			elif sun == 1 : l[idx] = 6; continue
	
		# create new columns
		self.X1.loc[:, 'weekday_sin'] = np.sin(l*2*np.pi/7)
		self.X1.loc[:, 'weekday_cos'] = np.cos(l*2*np.pi/7)

		# drop old columns
		col = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']
		self.X1 = self.X1.drop(columns = col)
		
		if VERBOSE : print("Transformed one-hot encodings in sin-cos weekdays & dropped one-hot encodings")
		
	def remove_outliers(self):
		isolation_forest = IsolationForest(n_jobs=-1)
		index = isolation_forest.fit_predict(np.append(self.X1, self.Y1, axis=1))
		to_remove = np.arange(0,len(index),1)[index==-1]
		self.X1 = self.X1.drop(index=to_remove)
		self.Y1 = self.Y1.drop(index=to_remove)
		if VERBOSE : print("removed " + len(to_remove)+ " outliers")

	
	def kernel_pca(self, n_features : int = 15, kernel = 'linear'):
		"""
		KernelPCA for feature selection
		DOC : https://scikit-learn.org/stable/modules/decomposition.html#kernel-pca
		"""
		kpca = KernelPCA(n_components=n_features, kernel=kernel, gamma=1/n_features, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None)
	
	def describe_features(self):
		"""
		Print a description of the features
		"""
		print(self.X1.describe)

	def get_features_names(self):
		"""
		Print the names of the different features
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

	def remove_correlation_features(self, th=0.9):
		cor = np.abs(np.corrcoef(self.X1_scaled, self.Y1.values, rowvar=False))
		upperCor = np.triu(cor, k=1)[:-1,:-1]						#k=1 to ignore the diagonal and [:-1,:-1] to ignore the correlation with the target
		stronglyCorrelated = np.argwhere(upperCor > th)
		mutualInfo = mutual_info_regression(self.X1_scaled, np.ravel(self.Y1))	#quite slow
		for pair in stronglyCorrelated:
			if VERBOSE : print("those features are highly corelated:", self.X1.columns.values[pair], "they have a correlation of", upperCor[pair[0],pair[1]] )
			indexToRemove = pair[np.argsort(mutualInfo[pair])[0]]	#desole Guss, je sais c'est pas lisible mais il est tard donc je m embete pas
			nameToRemove = self.X1.columns[indexToRemove]
			if VERBOSE : print("their mutual information with the target:", mutualInfo[pair])
			if VERBOSE : print(nameToRemove, "has the lowest mutual info with the target. I remove it")
			self.X1 = self.X1.drop(nameToRemove, axis=1)

	def predict_with_linear_regression(self):
		"""
		Fit a linear regressor using the training data and make a prediction of the test data
		"""
		linear_regressor = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
		linear_regressor.fit(self.X_train, self.Y_train)
		prediction = linear_regressor.predict(self.X_test)
		return prediction, linear_regressor.coef_

	def predict_with_lasso(self, max_iter : int = 1100, tol : float = 1e-4, warm_start : bool = False):
		"""
		Linear model trained with L1 prior as regularizer (aka the Lasso). 
		DOC : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso:
		"""
		lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, max_iter=max_iter, tol=tol, warm_start=warm_start, selection='random')
		# we can also use selection='cyclic' to loop over features sequentially
		lasso.fit(self.X_train, self.Y_train)
		prediction = lasso.predict(self.X_test)
		return prediction, lasso.coef_
	

	def get_grid_search_knn(self):
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
		
		gs.fit(self.X_train, self.Y_train)

		if VERBOSE:
			print("--- Grid search KNN ---")
			print("best parameters:", gs.best_params_)
			print("training score:", gs.best_score_)

		return gs

	def get_grid_search_mlp(self):
		scoring = {
			'NegMSE': 'neg_mean_squared_error', 
			'score_regression': metrics.make_scorer(score_regression, greater_is_better=True)
		}
		grid = {
			'hidden_layer_sizes': [(100,), (140,), (50,50,), (50,)],
			'activation': ['identity', 'logistic', 'tanh', 'relu'],
			'solver': ['adam'],
			'alpha': 10.0 ** -np.arange(1, 7),
			# DOC : alpha advised by https://scikit-learn.org/stable/modules/neural_networks_supervised.html
			'learning_rate': ['constant'],	#{‘constant’, ‘invscaling’, ‘adaptive’}
			'learning_rate_init': [0.1]
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
