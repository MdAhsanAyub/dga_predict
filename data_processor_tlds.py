__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Smith, Steven", "Yilmiz, Ibrahim",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"

# Importing the libraries
import numpy as np
import pandas as pd

#importing the data set
dataset = pd.read_excel('sample_production_data.xlsx')
X = dataset.iloc[:, 2:3]
Y = dataset.iloc[:, -1]

# One hot encoding of TLD features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.values[:, 0] = labelencoder_X.fit_transform(X.values[:, 0])
onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
X = onehotencoder_X.fit_transform(X).toarray()

# Generating the CSV file for further usuage.            
pd.DataFrame(X).to_csv("One_Hot_Encoded_TLDs.csv")