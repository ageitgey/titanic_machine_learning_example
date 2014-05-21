# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# We are going to create a feature from each name. There seems to be a correlation
# between the title of each person and whether or not they survived.  So let's
# turn the name column into just a number representing the person's title.
# No title = -1, "Master" = 0, "Mr" = 1, etc.
def parseName(name):
  out = -1
  names = ["Master.", "Mr.", "Dona.", "Miss.", "Mrs.", "Dr.", "Rev.", "Col.", "Ms.", "Capt.", "Mlle.", "Major.", "Mme."]
  for i, n in enumerate(names):
    if n in name:
      out = i
      break
  return out

# The data from Kaggle needs some cleaning
def cleanData(data):
  # If fare data is missing, replace it with the average from that class
  data.Fare = data.Fare.map(lambda x: np.nan if x==0 else x)
  classmeans = data.pivot_table('Fare', rows='Pclass', aggfunc='mean')
  data.Fare = data[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )

  # Turn names into a number representing titles
  data.Name = data.Name.map(lambda x: parseName(x))

  # Covert sex into a numberic value
  data.Sex = data.Sex.apply(lambda sex: 0 if sex == "male" else 1)

  return data


# Load training and test data sets, cleaning them in the process
train = cleanData(pd.read_csv("train.csv"))
test = cleanData(pd.read_csv("test.csv"))

# Pick out the four columns we care about and split the training set in features (X)
# and labels (y)
cols = ["Fare", "Pclass", "Sex", "Name"]
X = train[cols].values
y = train['Survived'].values

# To use an SVC, data needs to be scaled between [-1, 1] with a 0 mean.
# Scaling won't negatively affect any of the other classifiers.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create the classifiers
clf1 = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=1, random_state=0)
clf2 = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=1, random_state=0)
clf3 = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
clf4 = AdaBoostClassifier(n_estimators=500)
clf5 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
clf6 = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.2, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)

clfs = [clf1, clf2, clf3, clf4, clf5, clf6]

# Fit each classifier based on the training data
for clf in clfs:
  clf.fit(X, y)

# Now create features from the test set
X = test[cols].values
X = scaler.transform(X)

# For all 6 classifiers, predict outputs and save the probabilities of each prediction
predictions = []

for clf in clfs:
  predictions.append(clf.predict_proba(X))

# Now we have six sets of predictions in a list.  Average across all lists to create
# one average prediction across all classifiers.
# Note: There are smarter ways to do this, but this works fairly well.
p = np.mean(predictions, axis=0)

# Now we have one list where each element is a tuple of (prob_true, prob_false).
# Let's turn those into 0 or 1 based on the prob_true value.
p = map(lambda x: 0 if x[0] >= 0.5 else 1, p)

# Now we have a prediction for each item of 0 or 1. Just write the result to a
# csv file in the format that Kaggle wants
with open('predictions.csv', 'wb') as csvfile:
  w = csv.writer(csvfile)
  w.writerow(["PassengerId", "Survived"])

  for i in xrange(len(p)):
    w.writerow([test.PassengerId[i], p[i]])
