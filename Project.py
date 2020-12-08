# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# verbose
VERBOSE_LEVEL = 0

class Project:
  """
  Class used to be able to easily access different values and pass information between functions
  """
  def read_data(self,
    X1_file : str = "X1.csv",
    Y1_file : str = "Y1.csv"):

    self.X1 = pd.read_csv(X1_file)
    self.Y1 = pd.read_csv(Y1_file, header=None, names=['shares '])

  def normalizeData(data, scaler=StandardScaler()):
    return scaler.fit_transform(data)

  def score_f1(y_true, y_pred, th):
    return sklearn.metrics.f1_score(y_true > th, y_pred > th)

  def score_regression(y_true , y_pred):
    scores = [ score_f1(y_true, y_pred, th=th) for th in [500, 1400, 5000, 10000] ]
    return np.mean(scores)
