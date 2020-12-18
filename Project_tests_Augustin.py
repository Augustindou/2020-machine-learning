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
Comments on the project
	- As of python 3.5 (I think), you might see datatypes given explicitely (x : int = 3). This is only used for code clarity as python will not care about the explicit type given (x : int = 3.2 is perfectly valid and type(x) will return 'float').
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

class Project:
	"""
	Class used to be able to easily access different values and pass information between functions
	"""

	def __init__(self):
		# WIP
		self.read_data()
		self.split_data()
		self.normalize_data()
		# mettre les methodes de feature selection ici

		# ensuite on peut split nos data
		self.X_trainScaled, self.X_testScaled = self.normalize_data(self.X_train, self.X_test)

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
		Documentation : https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
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

	def predict_with_linear_regression(self):
		"""
		Fit a linear regressor using the training data and make a prediction of the test data
		"""
		linear_regressor = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
		linear_regressor.fit(self.X_train_scaled, self.Y_train)
		prediction = linear_regressor.predict(self.X_test_scaled)
		return prediction, linear_regressor.coef_

	def predict_with_lasso(self, max_iter : int = 1100, tol : float = 1e-4, warm_start : bool = False):
		"""
		Linear model trained with L1 prior as regularizer (aka the Lasso). 
		Documentation : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso:
		"""
		lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, max_iter=max_iter, tol=tol, warm_start=warm_start, selection='random')
		# we can also use selection='cyclic' to loop over features sequentially
		lasso.fit(self.X_train_scaled, self.Y_train)
		prediction = lasso.predict(self.X_test)
		return prediction, lasso.coef_
	
	def predict_with_knn(self, n_neighbors : int = 5, weights = 'uniform', algorithm = 'auto', leaf_size : int = 30, p : int = 2):
		"""
		Prediction with KNN
		Documentation : https://scikit-learn.org/stable/modules/neighbors.html#neighbors (important for undersanding the choice of algorithm to use)
		@param weights : {'uniform', 'distance'}
		@param algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
		? @Gauthier p pour manhattan_distance ou euclidean_distance ou autre pour minkowski dependant de p
		"""
		# other parameters for KNeighborsRegressor : metric='minkowski', metric_params=None
		knn = KNeighborsRegressor(n_neighbors, weights, algorithm, leaf_size, p, n_jobs=-1)
		knn.fit(self.X_train_scaled, self.Y_train)
		prediction = knn.predict(self.X_test)
		return prediction

	def predict_with_mlp(self):
		"""
		Prediction with MLP
		Documentation : https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
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
		mlp.fit(self.X_train_scaled, self.Y_train)
		prediction = mlp.predict(self.X_test)
		return prediction

	def get_grid_search_knn(self):
		"""
		Documentation : https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
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
		gs.fit(self.X_train_scaled, self.Y_train)

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
			'alpha': 10.0 ** -np.arange(1, 7) 
			# documentation : alpha advised by https://scikit-learn.org/stable/modules/neural_networks_supervised.html
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
		gs.fit(self.X_train_scaled, self.Y_train)
		
		if VERBOSE:
			print("--- Grid search MLP ---")
			print("best params:", gs.best_params_)
			print("training score:", gs.best_score_)

		return gs













# ------------------------------------------------------------------------------
# --------------------------- OLD CODE FROM GAUTHIER ---------------------------
# ------------------------------------------------------------------------------
"""
	Function fit a linearRegressor using the target and make a prediction based on a testing set
	@param:	trainingSet:[{array-like, sparse matrix} of shape (n_samples, n_features)] training data
			target: [array-like of shape (n_samples,) or (n_samples, n_targets)] targeted value corresponding to the trainingSet to trin the regressor
			testingSet: [array_like or sparse matrix, shape (n_samples, n_features)] input values of the trained regressor
	@returns: [array, shape (n_samples,)] predicted values and [ndarray of shape (n_features,) or (n_targets, n_features)] Estimated coefficients
"""
def predictWithLinearRegression(trainingSet, target, testingSet):
	linearRegressor = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
	return linearRegressor.fit(trainingSet, target).predict(testingSet), linearRegressor.coef_

"""
	From https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso:
	Linear Model trained with L1 prior as regularizer (aka the Lasso) The optimization objective for Lasso is:
	(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
	@param:	trainingSet:[{array-like, sparse matrix} of shape (n_samples, n_features)] training data
			target: [array-like of shape (n_samples,) or (n_samples, n_targets)] targeted value corresponding to the trainingSet to trin the regressor
			testingSet: [array_like or sparse matrix, shape (n_samples, n_features)] input values of the trained regressor
			max_iter: [int] The maximum number of iterations
			tol: [float] The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality
				and continues until it is smaller than tol.
			warm_start: [bool] When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
	@returns: [array, shape (n_samples,)] predicted values and [ndarray of shape (n_features,) or (n_targets, n_features)] Estimated coefficients
"""
def predictWithLasso(trainingSet, target, testingSet, max_iter=1100, tol=0.0001, warm_start=False):
	lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, max_iter=max_iter, tol=tol, warm_start=warm_start, selection='random')#selection='cyclic' to loop over features sequentially
	lasso.fit(trainingSet, target)
	return lasso.predict(testingSet), lasso.coef_


proj = Project()
LinearRegressionPrediction,_ = predictWithLinearRegression(proj.X_trainScaled, proj.Y_train, proj.X_trainScaled)
print("score by LinearRegression testing from learned data:", score_regression(proj.Y_train, LinearRegressionPrediction))	#0.48896528584814974
LassoPrediction,_ = predictWithLasso(proj.X_trainScaled, proj.Y_train, proj.X_trainScaled)
print("score by Lasso testing from learned data:", score_regression(proj.Y_train, LassoPrediction))	#0.488894300572261

LinearRegressionPrediction,_ = predictWithLinearRegression(proj.X_trainScaled, proj.Y_train, proj.X_testScaled)
print("score by LinearRegression:", score_regression(proj.Y_test, LinearRegressionPrediction))	#0.48243917542285
LassoPrediction,_ = predictWithLasso(proj.X_trainScaled, proj.Y_train, proj.X_testScaled)
print("score by Lasso testing:", score_regression(proj.Y_test, LassoPrediction))	#0.4834171812808842

A = np.corrcoef(proj.X_train, proj.Y_train, rowvar=False)
B = np.corrcoef(proj.X_trainScaled, proj.Y_train, rowvar=False)
print(A[1,2], B[1,2])

#/!\ preneur en temps mais beau et instructif sur les features donnant potentiellement les mêmes info.
#proj.plotCorrelationMatrix(filename="correlation_matNotNormalized.svg", normalize=False)
#proj.plotCorrelationMatrix()


# weights{‘uniform’, ‘distance’}; algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}; p pour manhattan_distance ou euclidean_distance ou autre pour minkowski dependant de p
def predictWithKNN(self, trainingSet, target, testingSet, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2):
	# absolument lire https://scikit-learn.org/stable/modules/neighbors.html#neighbors qui parle du choix de l'algo (pas auto) et des leaf pour des problèmes donnés!!
	knn = KNeighborsRegressor(n_neighbors, weights, algorithm, leaf_size, p, n_jobs=-1)#metric='minkowski', metric_params=None
	knn.fit(trainingSet, target)
	return knn.predict(testingSet)

def predictWithMLP(self, trainingSet, target, testingSet):
	mlp = MLPRegressor(
		hidden_layer_sizes=(100, ),	#ca ca fait que une layer de 100 neurones la...
		activation='relu',	#{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
		solver='adam',		#{‘lbfgs’, ‘sgd’, ‘adam’} #sgd lbfgs pour petits sets, adam robuste, sgd donne de meilleurs resultats si le learning rate est bien reglé
		alpha=0.0001,		#regularization term
		batch_size='auto',
		learning_rate='constant',	#{‘constant’, ‘invscaling’, ‘adaptive’}
		learning_rate_init=0.001,
		power_t=0.5,		#Only used when solver=’sgd’.
		max_iter=200,
		shuffle=True,		#Only used when solver=’sgd’ or ‘adam’.
		random_state=None,
		tol=0.0001,
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
	mlp.fit(trainingSet, target)
	return mlp.predict(testingSet)

def getGridSearchKNN(proj):
	scoring = {'NegMSE': 'neg_mean_squared_error', 'score_regression': metrics.make_scorer(proj.score_regression, greater_is_better=True)} #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
	# Setting refit='score_regression', refits an estimator on the whole dataset with the
	# parameter setting that has the best cross-validated score_regression score.
	gs = model_selection.GridSearchCV(KNeighborsRegressor(algorithm='auto', n_jobs=-1),
						param_grid={
							'n_neighbors': [3, 5, 7, 9, 13, 15, 20, 30],
							'weights': ['uniform', 'distance']
							},
						scoring=scoring, refit='score_regression', return_train_score=True, error_score=0, n_jobs=-1, verbose=3)
	gs.fit(proj.X_trainScaled, proj.Y_train)
	print("best params:", gs.best_params_)
	print("training score:", gs.best_score_)
	return gs

#gsKNN = getGridSearchKNN(proj)
KNNprediction = gsKNN.predict(proj.X_testScaled)	#best params: {'n_neighbors': 13, 'weights': 'uniform'}; training score: 0.5023354439139947
print("score by LinearRegression:", proj.score_regression(proj.Y_test, KNNprediction))	#0.4874824203078591

def getGridSearchMLP(proj):
	scoring = {'NegMSE': 'neg_mean_squared_error', 'score_regression': metrics.make_scorer(proj.score_regression, greater_is_better=True)}

	gs = model_selection.GridSearchCV(MLPRegressor(),
		param_grid={
			'hidden_layer_sizes': [(100,), (140,), (50,50,), (50,)],
			'activation': ['identity', 'logistic', 'tanh', 'relu'],
			'solver': ['adam'],
			'alpha': 10.0 ** -np.arange(1, 7) # adviced by https://scikit-learn.org/stable/modules/neural_networks_supervised.html
			# TODO
		},
		scoring=scoring, refit='score_regression', return_train_score=True, error_score=0, n_jobs=-1, verbose=3)
	gs.fit(proj.X_trainScaled, proj.Y_train)
	print("best params:", gs.best_params_)
	print("training score:", gs.best_score_)
	return gs

#gsMLP = getGridSearchMLP(proj)
#MLPprediction = gsMLP.predict(proj.X_testScaled)
#print(f"score by LinearRegression: {proj.score_regression(proj.Y_test, MLPprediction)}")

"""
Trash mais que je veux pas supprimer quand même ^^ :
def printMostCorrFeat(self, magnitude=0.80):
	X1Normalized = normalize_data_as_panda(self.X1)
	AllCorr = X1Normalized.corr().unstack()
	stronglyCorr = AllCorr[abs(AllCorr) >= magnitude]
	print(stronglyCorr)

#contenu de kfold #Ha bah c'est fait dans grid search avec cv=None
donnee = np.arange(0,10,1)
kf = model_selection.KFold(n_splits=5)#, shuffle=True)
trainTest_index = np.array([indexs for indexs in kf.split(donnee)])
def getSplit(i):
	return donnee[trainTest_index[i%len(trainTest_index),0]]
"""

"""
lecture pour la prochaine fois:
https://scikit-learn.org/stable/modules/grid_search.html#grid-search
https://scikit-learn.org/stable/modules/neighbors.html#neighbors -> le bas est potentiellement très important pour la features selection!

https://scikit-learn.org/stable/modules/neural_networks_supervised.html -> voir les tips en bas, dit ce qu'il faut mettre dans grid_search + dit qu'on triche en fittant le scaller sur le trainSet et testSet.
"""

"""
shuffle ou pas??
If the data ordering is not arbitrary (e.g. samples with the same class label are contiguous), shuffling it first may be essential to get a meaningful cross-validation result.
However, the opposite may be true if the samples are not independently and identically distributed. For example, if samples correspond to news articles, and are ordered by their time of publication,
then shuffling the data will likely lead to a model that is overfit and an inflated validation score: it will be tested on samples that are artificially similar (close in time) to training samples.
"""
