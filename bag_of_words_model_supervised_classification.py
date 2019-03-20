__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Smith, Steven", "Yilmiz, Ibrahim",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"

# Importing the libraries
import pandas as pd

#importing the data set
dataset = pd.read_csv('sample_production_data.csv', sep='\t')
print(dataset.head())
# Processing the domain names (text)
import re
corpus = []
for i in range(0,97734):
    domains = re.sub('[.]', ' ', dataset['Domain'][i]);
    domains = domains.lower()
    domains = domains.split()
    domains = ' '.join(domains)
    corpus.append(domains)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values

# Spliting the dataset into the Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Fitting the dataset into the Training set (Implementing Naive Bayes)
from sklearn.naive_bayes import GaussianNB
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = naive_bayes_classifier.predict(X_test)

# Making the cufusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

#	[[ 3545 6552]
#	[ 0 14337]]