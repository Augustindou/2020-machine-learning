# imports
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_regression

#sklearn.preprocessing.normalize ??? je sais pas si il faut

# verbose
VERBOSE_LEVEL = 0

class Project:
	"""
	Class used to be able to easily access different values and pass information between functions
	"""
	def __init__(self):
		self.read_data()
		#mettre les methodes de feature selection ici


		self.splitData()
		self.X1Scaled,_ = self.normalizeData(self.X1, self.X1)

		#ensuite on peut split nos data
		self.x_trainScaled, self.x_testScaled = self.normalizeData(self.x_train, self.x_test)

	def read_data(self,
		X1_file : str = "X1.csv",
		Y1_file : str = "Y1.csv") -> None:

		self.X1 = pd.read_csv(X1_file)
		self.Y1 = pd.read_csv(Y1_file, header=None, names=['shares '])

	def describeFeatures(self):
		print(self.X1.describe)

	def getFeatureNames(self):
		print(self.X1.columns)

	def splitData(self):
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X1, self.Y1, test_size=0.20)

	def normalizeData(self, trainingSet, testingSet):
		scaler = StandardScaler()
		scaler.fit(trainingSet)
		trainScaled = scaler.transform(trainingSet)
		testScaled = scaler.transform(testingSet)
		return trainScaled, testScaled

	def score_f1(self, y_true, y_pred, th):
		return sklearn.metrics.f1_score(y_true > th, y_pred > th)

	def score_regression(self, y_true , y_pred):
		scores = [ self.score_f1(y_true, y_pred, th=th) for th in [500, 1400, 5000, 10000] ]
		return np.mean(scores)

	"""
		Function to normalise a panda DataFrame.
		@param:	data: a panda [DataFrame] to normalize
				scaler: the scaller to use
		@returns: a normalized panda [DataFrame]
	"""
	def normalizeDataAsPanda(self, data, scaler=StandardScaler()):
		normalizedData = scaler.fit_transform(data)
		dataNormalizedPanda = pd.DataFrame(data=normalizedData, index=data.index, columns=data.columns)
		assert np.all(normalizedData == dataNormalizedPanda.values)
		return dataNormalizedPanda

	def plotCorrelationMatrix(self, filename="correlation_mat.svg", normalize=True):
		if normalize:
			X1Normalized = normalizeDataAsPanda(self.X1)
			correlation_mat = X1Normalized.join(self.Y1).corr()
		else :
			correlation_mat = self.X1.join(self.Y1).corr()
		plt.subplots(figsize=(25,20))
		sns.heatmap(correlation_mat, annot = False)
		plt.savefig(filename)

	"""
		Function that removes the features that are overly linearly related with each other (too corelated).
		For each pair of two too corelated features, theone that has the lowest mutual information with the target is removed.
		@param:	th: [float] thresshold over wich the correlation between two features is supposed to be too high
	"""
	def removeCorrFeatures(self, th=0.9):
		cor = np.abs(np.corrcoef(self.X1Scaled, self.Y1.values, rowvar=False))
		upperCor = np.triu(cor, k=1)[:-1,:-1]						#k=1 to ignore the diagonal and [:-1,:-1] to ignore the correlation with the target
		stronglyCorrelated = np.argwhere(upperCor > th)
		mutualInfo = mutual_info_regression(self.X1Scaled, np.ravel(self.Y1))	#takes a bit of time
		for pair in stronglyCorrelated:
			print("those features are highly corelated:", self.X1.columns.values[pair], "they have a correlation of", upperCor[pair[0],pair[1]] )
			indexToRemove = pair[np.argsort(mutualInfo[pair])[0]]	#desole Guss, je sais c'est pas lisible mais il est tard donc je m embete pas
			nameToRemove = self.X1.columns[indexToRemove]
			print("their mutual information with the target:", mutualInfo[pair])
			print(nameToRemove, "has the lowest mutual info with the target. I remove it")
			self.X1 = self.X1.drop(nameToRemove, axis=1)
		# self.splitData()
		# self.X1Scaled,_ = self.normalizeData(self.X1, self.X1)
		# self.x_trainScaled, self.x_testScaled = self.normalizeData(self.x_train, self.x_test)



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
proj.removeCorrFeatures()
LinearRegressioPrediction,_ = predictWithLinearRegression(proj.x_trainScaled, proj.y_train, proj.x_trainScaled)
print("score by LinearRegression testing from learned data:", proj.score_regression(proj.y_train, LinearRegressioPrediction))	#0.48896528584814974
LassoPrediction,_ = predictWithLasso(proj.x_trainScaled, proj.y_train, proj.x_trainScaled)
print("score by Lasso testing from learned data:", proj.score_regression(proj.y_train, LassoPrediction))	#0.488894300572261

LinearRegressioPrediction,_ = predictWithLinearRegression(proj.x_trainScaled, proj.y_train, proj.x_testScaled)
print("score by LinearRegression:", proj.score_regression(proj.y_test, LinearRegressioPrediction))	#0.48243917542285
LassoPrediction,_ = predictWithLasso(proj.x_trainScaled, proj.y_train, proj.x_testScaled)
print("score by Lasso testing:", proj.score_regression(proj.y_test, LassoPrediction))	#0.4834171812808842

A = np.corrcoef(proj.x_train, proj.y_train, rowvar=False)
B = np.corrcoef(proj.x_trainScaled, proj.y_train, rowvar=False)
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
	scoring = {'NegMSE': 'neg_mean_squared_error', 'score_regression': make_scorer(proj.score_regression, greater_is_better=True)} #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
	# Setting refit='score_regression', refits an estimator on the whole dataset with the
	# parameter setting that has the best cross-validated score_regression score.
	gs = GridSearchCV(KNeighborsRegressor(algorithm='auto', n_jobs=-1),
						param_grid={
							'n_neighbors': [3, 5, 7, 9, 13, 15, 20, 30],
							'weights': ['uniform', 'distance']
							},
						scoring=scoring, refit='score_regression', return_train_score=True, error_score=0, n_jobs=-1, verbose=3)
	gs.fit(proj.x_trainScaled, proj.y_train)
	print("best params:", gs.best_params_)
	print("training score:", gs.best_score_)
	return gs

#gsKNN = getGridSearchKNN(proj)
#KNNprediction = gsKNN.predict(proj.x_testScaled)	#best params: {'n_neighbors': 13, 'weights': 'uniform'}; training score: 0.5023354439139947
#print("score by KNN:", proj.score_regression(proj.y_test, KNNprediction))	#0.4874824203078591

def getGridSearchMLP(proj):
	scoring = {'NegMSE': 'neg_mean_squared_error', 'score_regression': make_scorer(proj.score_regression, greater_is_better=True)}
	gs = GridSearchCV(MLPRegressor(),
		param_grid={
			'hidden_layer_sizes': [(100,), (140,), (50,50,), (50,)],
			'activation': ['identity', 'logistic', 'tanh', 'relu'],
			'solver': ['adam'],
			'alpha': 10.0 ** -np.arange(1, 7), # adviced by https://scikit-learn.org/stable/modules/neural_networks_supervised.html
			'learning_rate': ['constant'],	#{‘constant’, ‘invscaling’, ‘adaptive’}
			'learning_rate_init': [0.1]
			# TODO
		},
		scoring=scoring, refit='score_regression', return_train_score=True, error_score=0, n_jobs=-1, verbose=3)
	gs.fit(proj.x_trainScaled, proj.y_train)
	print("best params:", gs.best_params_)
	print("training score:", gs.best_score_)
	return gs

#gsMLP = getGridSearchMLP(proj)
#MLPprediction = gsMLP.predict(proj.x_testScaled)
#print(f"score by MLP: {proj.score_regression(proj.y_test, MLPprediction)}")

"""
Trash mais que je veux pas supprimer quand même ^^ :
def printMostCorrFeat(self, magnitude=0.80):
	X1Normalized = self.normalizeDataAsPanda(self.X1)
	AllCorr = X1Normalized.corr().unstack()
	stronglyCorr = AllCorr[abs(AllCorr) >= magnitude]
	print(stronglyCorr)

#contenu de kfold #Ha bah c'est fait dans grid search avec cv=None
donnee = np.arange(0,10,1)
kf = KFold(n_splits=5)#, shuffle=True)
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



B = np.arange(1,26,1)
A = np.reshape(B, (5,5))
A = np.triu(A, k=1)
A

stronglyCorrelated = np.argwhere(A > 12)
for pair in stronglyCorrelated:
	print(A[pair[0],pair[1]])
	print("those features are highly corelated:", A[0][pair])
