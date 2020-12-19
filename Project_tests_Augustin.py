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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import KernelPCA, PCA

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
		if VERBOSE : print("\n--- Preprocessing ---")
		self.read_data()
		# put days_one_hot_to_sin_cos() before normalize if we want to have a normal one-hot encoding
		self.days_one_hot_to_sin_cos()
		self.normalize_data()
		self.remove_outliers()

		if VERBOSE : print("\n--- Feature Selection ---")
		self.remove_correlation_features()
		self.pca() # pca ? kernel_pca ?
		# self.kernel_pca()

		if VERBOSE : print("\n--- Splitting data ---")
		self.split_data()

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

		if hasattr(self, 'pca_X1'):
			self.pca_X_train, self.pca_X_test, self.pca_Y_train, self.pca_Y_test = model_selection.train_test_split(self.pca_X1, self.Y1, test_size=test_size)
		if hasattr(self, 'kpca_X1'):
			self.kpca_X_train, self.kpca_X_test, self.kpca_Y_train, self.kpca_Y_test = model_selection.train_test_split(self.kpca_X1, self.Y1, test_size=test_size)

		if VERBOSE : print(f"Data has been split between train and test, with {test_size*100}% of the data used as testing data")

	def normalize_data(self):
		"""
		Normalize the data 
		DOC : https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
		"""
		scaler = StandardScaler()
		self.X1 = pd.DataFrame(data = scaler.fit_transform(self.X1), index=self.X1.index, columns=self.X1.columns)
		if VERBOSE : print("X1 has been normalized")
		
	def remove_outliers(self):
		if VERBOSE : print("Transformed one-hot encodings in sin-cos weekdays & dropped one-hot encodings")
		
		isolation_forest = IsolationForest(n_jobs=-1)
		index = isolation_forest.fit_predict(np.append(self.X1, self.Y1, axis=1))
		to_remove = np.arange(0,len(index),1)[index==-1]
		self.X1 = self.X1.drop(index=to_remove)
		self.Y1 = self.Y1.drop(index=to_remove)
		if VERBOSE : print(f"removed {len(to_remove)} outliers")

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

		l *= 2*np.pi/7
		self.X1.loc[:, 'weekday_sin'] = np.sin(l)
		self.X1.loc[:, 'weekday_cos'] = np.cos(l)

		# drop old columns
		col = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']
		self.X1 = self.X1.drop(columns = col)
	
	def pca(self, n_features : int = 15):
		"""
		PCA for feature selection
		DOC : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
		"""
		n_features_start = len(self.X1.columns)
		pca = PCA(n_components=n_features)
		self.pca_X1 = pd.DataFrame(pca.fit_transform(self.X1))
		if VERBOSE : print(f"PCA used on X1 : from {n_features_start} to {n_features} features")

	def kernel_pca(self, n_features : int = 15, kernel='linear'):
		"""
		KernelPCA for feature selection
		DOC : https://scikit-learn.org/stable/modules/decomposition.html#kernel-pca
		"""
		n_features_start = len(self.X1.columns)
		kpca = KernelPCA(n_components=n_features, kernel=kernel, gamma=1/n_features, n_jobs=-1)
		self.kpca_X1 = pd.DataFrame(kpca.fit_transform(self.X1))
		if VERBOSE : print(f"KernelPCA used on X1 : from {n_features_start} to {n_features} features")
	
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
		cor = np.abs(np.corrcoef(self.X1, self.Y1.values, rowvar=False))
		upper_cor = np.triu(cor, k=1)[:-1,:-1]						#k=1 to ignore the diagonal and [:-1,:-1] to ignore the correlation with the target
		strongly_correlated = np.argwhere(upper_cor > th)
		mutual_info = mutual_info_regression(self.X1, np.ravel(self.Y1))	#quite slow
		for pair in strongly_correlated:
			if VERBOSE : print("those features are highly corelated:", self.X1.columns.values[pair], "they have a correlation of", upper_cor[pair[0],pair[1]] )
			index_to_remove = pair[np.argsort(mutual_info[pair])[0]]
			name_to_remove = self.X1.columns[index_to_remove]
			if VERBOSE : print("their mutual information with the target:", mutual_info[pair])
			if VERBOSE : print(name_to_remove, "has the lowest mutual info with the target. I remove it")
			self.X1 = self.X1.drop(name_to_remove, axis=1)

	def predict_with_linear_regression(self, preprocessing = 'default'):
		"""
		Fit a linear regressor using the training data and make a prediction of the test data
		"""
		linear_regressor = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
		if preprocessing == 'pca':
			linear_regressor.fit(self.pca_X_train, self.pca_Y_train)
			prediction = linear_regressor.predict(self.pca_X_test)
		elif preprocessing == 'kpca':
			linear_regressor.fit(self.kpca_X_train, self.kpca_Y_train)
			prediction = linear_regressor.predict(self.kpca_X_test)
		else:
			linear_regressor.fit(self.X_train, self.Y_train)
			prediction = linear_regressor.predict(self.X_test)
		return prediction, linear_regressor.coef_

	def predict_with_lasso(self, max_iter : int = 1100, tol : float = 1e-4, warm_start : bool = False, preprocessing = 'default'):
		"""
		Linear model trained with L1 prior as regularizer (aka the Lasso). 
		DOC : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso:
		"""
		lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, max_iter=max_iter, tol=tol, warm_start=warm_start, selection='random')
		# we can also use selection='cyclic' to loop over features sequentially
		if preprocessing == 'pca':
			lasso.fit(self.pca_X_train, self.pca_Y_train)
			prediction = lasso.predict(self.pca_X_test)
		elif preprocessing == 'kpca':
			lasso.fit(self.kpca_X_train, self.kpca_Y_train)
			prediction = lasso.predict(self.kpca_X_test)
		else:
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
			'n_neighbors': [13, 14, 15, 16],
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

		if VERBOSE :
			print("--- Grid search KNN ---")
			print(f"best parameters : {gs.best_params_}")
			print(f"training score (on trained data) : {gs.best_score_}")
			print(f"score on test data : {gs.score(self.X_test, self.Y_test)}")

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
		
		if VERBOSE :
			print("--- Grid search MLP ---")
			print("best params:", gs.best_params_)
			print("training score (on trained data):", gs.best_score_)

		return gs

	def get_grid_search_etr(self):
		scoring = {
			'NegMSE': 'neg_mean_squared_error', 
			'score_regression': metrics.make_scorer(score_regression, greater_is_better=True)
		}
		grid = {
			'n_estimators' : [80, 100, 120, 150],
			'max_features' : ['auto', 'sqrt', 'log2']
		}
		
		gs = model_selection.GridSearchCV(
			ExtraTreesRegressor(n_jobs=-1),
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

	#following code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
	#param_as_abscice is the string representing a hyper-param -> exemple: 'n_estimators'
	def plot_grid_search_perf(self, scoring, gs, param_as_abscice):
		plt.figure(figsize=(13, 13))
		plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

		plt.xlabel(param_as_abscice)
		plt.ylabel("Score")

		ax = plt.gca()
		ax.set_xlim(0, 402)
		ax.set_ylim(0.73, 1)

		# Get the regular numpy array from the MaskedArray
		X_axis = np.array(gs[param_as_abscice].data, dtype=float)

		for scorer, color in zip(sorted(scoring), ['g', 'k']):
			for sample, style in (('train', '--'), ('test', '-')):
				sample_score_mean = gs['mean_%s_%s' % (sample, scorer)]
				sample_score_std = gs['std_%s_%s' % (sample, scorer)]
				ax.fill_between(X_axis, sample_score_mean - sample_score_std,
								sample_score_mean + sample_score_std,
								alpha=0.1 if sample == 'test' else 0, color=color)
				ax.plot(X_axis, sample_score_mean, style, color=color,
						alpha=1 if sample == 'test' else 0.7,
						label="%s (%s)" % (scorer, sample))

			best_index = np.nonzero(gs['rank_test_%s' % scorer] == 1)[0][0]
			best_score = gs['mean_test_%s' % scorer][best_index]

			# Plot a dotted vertical line at the best score for that scorer marked by x
			ax.plot([X_axis[best_index], ] * 2, [0, best_score],
					linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

			# Annotate the best score for that scorer
			ax.annotate("%0.2f" % best_score,
						(X_axis[best_index], best_score + 0.005))

		plt.legend(loc="best")
		plt.grid(False)
		plt.show()




scoring = {
	'NegMSE': 'neg_mean_squared_error', 
	'score_regression': metrics.make_scorer(score_regression, greater_is_better=True)
}

# p = Project()

# print("\n--- Normal ---")
# # linear regression scaled
# prediction,_ = p.predict_with_linear_regression()
# print("score by LinearRegression:", score_regression(p.Y_test, prediction)) # 0.48243917542285	

# # lasso scaled
# prediction,_ = p.predict_with_lasso()
# print("score by Lasso testing:", score_regression(p.Y_test, prediction))	# 0.4834171812808842

# print("\n--- PCA ---")
# # linear regression scaled
# prediction,_ = p.predict_with_linear_regression(preprocessing = "pca")
# print("score by LinearRegression:", score_regression(p.pca_Y_test, prediction)) 

# # lasso scaled
# prediction,_ = p.predict_with_lasso(preprocessing = "pca")
# print("score by Lasso testing:", score_regression(p.pca_Y_test, prediction))	

# print("\n--- Kernel PCA ---")
# # linear regression scaled
# prediction,_ = p.predict_with_linear_regression(preprocessing = "kpca")
# print("score by LinearRegression:", score_regression(p.kpca_Y_test, prediction)) 

# # lasso scaled
# prediction,_ = p.predict_with_lasso(preprocessing = "kpca")
# print("score by Lasso testing:", score_regression(p.kpca_Y_test, prediction))	

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
