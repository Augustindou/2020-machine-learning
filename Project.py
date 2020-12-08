# imports
import pandas as pd
import numpy as np

# verbose
VERBOSE_LEVEL = 0

class Project:
  """
  Class used to be able to easily access different values and pass information between functions
  """
  def read_data(self,
    X1_file : str = "X1.csv",
    Y1_file : str = "Y1.csv",):
    
    self.X1 = pd.read_csv(X1_file)
    self.Y1 = pd.read_csv(Y1_file)