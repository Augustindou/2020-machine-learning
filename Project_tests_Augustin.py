# imports
from numpy.core.fromnumeric import sort
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

# constants
n_features_pca_kpca = 30

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
		if VERBOSE : print("\n--- Reading the files ---")
		self.read_data()

		if VERBOSE : print("\n--- Preprocessing ---")
		# put days_one_hot_to_sin_cos() before normalize if we want to have a normal one-hot encoding
		self.X1 = self.days_one_hot_to_sin_cos(self.X1)
		self.X2 = self.days_one_hot_to_sin_cos(self.X2)
		self.split_data()
		self.normalize_data()
		self.remove_outliers_on_train_set()

		if VERBOSE : print("\n--- Feature Selection ---")
		self.remove_correlation_features()
		# self.pca()
		# self.kernel_pca()


	def read_data(self,
		X1_file : str = "X1.csv",
		Y1_file : str = "Y1.csv",
		X2_file : str = "X2.csv") -> None:
		"""
		read the data from the input files
		"""

		# load the input and output files
		self.X1 : pd.DataFrame = pd.read_csv(X1_file)
		self.Y1 : pd.DataFrame = pd.read_csv(Y1_file, header=None, names=['shares '])
		self.X2 : pd.DataFrame = pd.read_csv(X2_file)
		if VERBOSE : print(f"Data has been retrieved from files '{X1_file}' and '{Y1_file}' and '{X2_file}'")

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
		scaler = StandardScaler()
		self.X_train = pd.DataFrame(data = scaler.fit_transform(self.X_train), columns=self.X1.columns)
		self.X_test = pd.DataFrame(data = scaler.transform(self.X_test), columns=self.X1.columns)
		self.X2 = pd.DataFrame(data = scaler.transform(self.X2), index=self.X2.index, columns=self.X2.columns)
		if VERBOSE : print("X_train, X_test and X2 has been normalized")

	def remove_outliers_on_train_set(self):
		isolation_forest = IsolationForest(n_jobs=-1)
		index = isolation_forest.fit_predict(np.append(self.X_train, self.Y_train, axis=1))
		to_remove = self.X_train.index[index==-1]
		self.X_train = self.X_train.drop(index=to_remove)
		to_remove = self.Y_train.index[index==-1]
		self.Y_train = self.Y_train.drop(index=to_remove)
		if VERBOSE : print(f"removed {len(to_remove)} outliers")

	def days_one_hot_to_sin_cos(self, data):
		# transform one-hot into list : monday = 0, tuesday = 1, ...
		l = np.zeros(len(data['weekday_is_monday']))
		for idx, mon, tue, wed, thu, fri, sat, sun in zip(range(len(l)),
			data['weekday_is_monday'],
			data['weekday_is_tuesday'],
			data['weekday_is_wednesday'],
			data['weekday_is_thursday'],
			data['weekday_is_friday'],
			data['weekday_is_saturday'],
			data['weekday_is_sunday']):

			if   mon == 1 : l[idx] = 0; continue
			elif tue == 1 : l[idx] = 1; continue
			elif wed == 1 : l[idx] = 2; continue
			elif thu == 1 : l[idx] = 3; continue
			elif fri == 1 : l[idx] = 4; continue
			elif sat == 1 : l[idx] = 5; continue
			elif sun == 1 : l[idx] = 6; continue

		l *= 2*np.pi/7
		data.loc[:, 'weekday_sin'] = np.sin(l)
		data.loc[:, 'weekday_cos'] = np.cos(l)

		# drop old columns
		col = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']
		data = data.drop(columns = col)

		if VERBOSE : print("Transformed one-hot encodings in sin-cos weekdays & dropped one-hot encodings")
		return data

	def pca(self, n_features : int = n_features_pca_kpca):
		"""
		PCA for feature selection
		DOC : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
		"""
		n_features_start = len(self.X1.columns)
		pca = PCA(n_components=n_features)
		self.X_train = pd.DataFrame(pca.fit_transform(self.X_train))
		self.X_test = pd.DataFrame(pca.transform(self.X_test))
		self.X2 = pd.DataFrame(pca.fit_transform(self.X2))
		if VERBOSE : print(f"PCA used on X_train : from {n_features_start} to {n_features} features")

	def kernel_pca(self, n_features : int = n_features_pca_kpca, kernel='linear'):
		"""
		KernelPCA for feature selection
		DOC : https://scikit-learn.org/stable/modules/decomposition.html#kernel-pca
		"""
		n_features_start = len(self.X1.columns)
		kpca = KernelPCA(n_components=n_features, kernel=kernel, gamma=1/n_features, n_jobs=-1)
		self.X1 = pd.DataFrame(kpca.fit_transform(self.X_train))
		self.X_test = pd.DataFrame(kpca.transform(self.X_test))
		self.X2 = pd.DataFrame(kpca.fit_transform(self.X2))
		if VERBOSE : print(f"KernelPCA used on X_train : from {n_features_start} to {n_features} features")

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

	def remove_correlation_features(self, th=0.85):
		cor = np.abs(np.corrcoef(self.X_train, self.Y_train.values, rowvar=False))
		upper_cor = np.triu(cor, k=1)[:-1,:-1]						#k=1 to ignore the diagonal and [:-1,:-1] to ignore the correlation with the target
		strongly_correlated = np.argwhere(upper_cor > th)
		mutual_info = mutual_info_regression(self.X_train, np.ravel(self.Y_train))	#quite slow
		set_to_remove = set()
		for pair in strongly_correlated:
			if VERBOSE : print("these features are highly corelated:", self.X_train.columns.values[pair], "they have a correlation of", upper_cor[pair[0],pair[1]] )
			if (pair[0] not in set_to_remove) and (pair[1] not in set_to_remove):
				index_to_remove = pair[np.argsort(mutual_info[pair])[0]]
				name_to_remove = self.X_train.columns[index_to_remove]
				if VERBOSE : print("their mutual information with the target:", mutual_info[pair])
				if VERBOSE : print(name_to_remove, "has the lowest mutual info with the target. I will remove it")
				set_to_remove.add(name_to_remove)
		self.X_train = self.X_train.drop(list(set_to_remove), axis=1)
		self.X2 = self.X2.drop(list(set_to_remove), axis=1)
		if VERBOSE : print(f"removed {len(set_to_remove)} correlated features")

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
			'score_regression': metrics.make_scorer(score_regression, greater_is_better=True)
		}
		grid = {
			'n_neighbors': [4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25, 30, 40],
			'weights': ['uniform']# , 'distance']
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
			'score_regression': metrics.make_scorer(score_regression, greater_is_better=True)
		}
		grid = {
			'hidden_layer_sizes': [(25, 50, 25,)],
			# 'hidden_layer_sizes': [(15,), (50,), (100,), (50, 50,), (25, 50, 25,), (25, 25, 25, 25,), (50, 50, 50), (25, 50, 50, 25)],
			'activation': ['relu'], # {'identity', 'logistic', 'tanh', 'relu'}
			'solver': ['adam'],
			'alpha': [1e-5], # 10.0 ** -np.arange(1, 7),
			# DOC : alpha advised by https://scikit-learn.org/stable/modules/neural_networks_supervised.html
			'learning_rate': ['constant'], # {'constant', 'invscaling', 'adaptive'}
			'learning_rate_init': 10.0 ** -np.arange(0, 5), # 0.0001 default, 0.001 for other graphs
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
	
		gs.fit(self.X_train, self.Y_train.values.ravel())
		
		if VERBOSE :
			print("--- Grid search MLP ---")
			print("best params:", gs.best_params_)
			print("training score (on trained data):", gs.best_score_)

		return gs

	def get_grid_search_etr(self):
		scoring = {
			'score_regression': metrics.make_scorer(score_regression, greater_is_better=True)
		}
		grid = {
			'n_estimators' : [10, 50, 80, 100, 120, 150, 200, 300],
			'max_features' : ['auto'] #, 'sqrt', 'log2']
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
	
		gs.fit(self.X_train, self.Y_train.values.ravel())
		
		if VERBOSE:
			print("--- Grid search MLP ---")
			print("best params:", gs.best_params_)
			print("training score:", gs.best_score_)

		return gs

	#following code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
	#param_as_abscice is the string representing a hyper-param -> exemple: 'n_estimators'
	def plot_grid_search_perf(self, scoring, gs, param_as_abscice = 'n_neighbors', title = ''):
		# ! only use with 1 changing variable in the grid, no time to make a more robust version :'(
		plt.figure(figsize=(13, 13))
		plt.title(title, fontsize=16)

		plt.xlabel(param_as_abscice)
		plt.ylabel("Score")

		ax = plt.gca()
		# ax.set_xlim(0, 402)
		# ax.set_ylim(0.73, 1)

		# Get the regular numpy array from the MaskedArray
		if param_as_abscice == 'learning_rate_init':
			X_axis = np.arange(len(gs.param_grid[param_as_abscice]))
			xticks = np.array(gs.param_grid[param_as_abscice])
			plt.xticks(X_axis, xticks)
		elif type(gs.param_grid[param_as_abscice][0]) == int or type(gs.param_grid[param_as_abscice][0]) == float :
			X_axis = np.array(gs.param_grid[param_as_abscice])
		else:
			X_axis = np.arange(len(gs.param_grid[param_as_abscice]))
			xticks = np.array(gs.param_grid[param_as_abscice])
			plt.xticks(X_axis, xticks)

		scorer = 'score_regression'
		color = 'b'
		for sample, style in (('test', '-'),): #('train', '--'),
			sample_score_mean = gs.cv_results_['mean_%s_%s' % (sample, scorer)]
			sample_score_std = gs.cv_results_['std_%s_%s' % (sample, scorer)]
			ax.fill_between(X_axis, sample_score_mean - sample_score_std,
							sample_score_mean + sample_score_std,
							alpha=0.1 if sample == 'test' else 0, color=color)
			ax.plot(X_axis, sample_score_mean, style, color=color,
					alpha=1 if sample == 'test' else 0.7,
					label="%s (%s)" % (scorer, "5-Fold test" if sample == 'test' else sample))

			best_index = np.nonzero(gs.cv_results_['rank_test_%s' % scorer] == 1)[0][0]
			best_score = gs.cv_results_['mean_test_%s' % scorer][best_index]

			# Plot an X on the best score
			ax.plot([X_axis[best_index], ], [best_score],
					linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)


			# Annotate the best score for that scorer
			ax.annotate("%0.3f" % best_score,
						(X_axis[best_index], best_score + 0.005))

		# plot a line for the test score of the best model
		test_score = gs.score(self.X_test, self.Y_test)
		ax.plot([X_axis[0], X_axis[-1]], [test_score, test_score], 
		    linestyle='dotted', color='g', markeredgewidth=3, ms=8, 
				label="best score on test data : %0.3f" % test_score)

		plt.legend(loc="best")
		plt.grid(False)
		plt.show()

scoring = {
	'score_regression': metrics.make_scorer(score_regression, greater_is_better=True)
}

# p = Project(); gs = p.get_grid_search_knn(); p.plot_grid_search_perf(scoring, gs, 'n_neighbors')
# p = Project(); gs = p.get_grid_search_mlp(); p.plot_grid_search_perf(scoring, gs, 'hidden_layer_sizes')
# p = Project(); gs = p.get_grid_search_mlp(); p.plot_grid_search_perf(scoring, gs, 'learning_rate_init')
# p = Project(); gs = p.get_grid_search_etr(); p.plot_grid_search_perf(scoring, gs, 'n_estimators')


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
