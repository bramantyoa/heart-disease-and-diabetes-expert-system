# select best features using several algorithms

# pandas is required
import pandas as pd

# read data
# jantung
jantung_df = pd.read_csv("data/jantung.csv", header=0, sep=";")
# diabetes
diabetes_df = pd.read_csv("data/diabetes.csv", header=0)

# 
# using Extra Trees Classifier
# 
from sklearn.ensemble import ExtraTreesClassifier

