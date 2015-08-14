import pandas
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import numpy as np
from tsne import bh_sne
import matplotlib.pyplot as plt

# https://www.dataquest.io/mission/74

# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pandas.read_csv("train.csv")

# Print some statistics of the dataframe.
print(titanic.describe())

# Replace all the missing values in the Age column of titanic
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Find all the unique genders -- the column appears to contain only male and female.
# Replace categorical data with numerical.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

tsne = bh_sne(titanic[predictors].astype('float64'), perplexity = 30, theta = 0.15)

plt.figure(figsize=(15, 5))
plt.scatter(tsne[:, 0], tsne[:, 1], c = titanic["Survived"])
plt.show()
