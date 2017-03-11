import matplotlib.pyplot as plt;
from sklearn import datasets;
from sklearn import svm;
import pandas as pd;
from sklearn.cross_validation import train_test_split;
from numpy import ravel;

#Variable for changing value of 'C'
C_var = 1;

#import of data
data = pd.read_csv('C:/Users/taylor.blank.THEBLANKFAMILY/Desktop/AP3/breast-cancer-wisconsin.csv');

#Selecting columns of data to include as predictors
X = data[['clump_thickness', 'Uniformity_cell_size', 'Uniformity_cell_shape', 'Marginal_adhesion', 'Single_epithelial_size', 'Bare_nuclei', 'Bland_chromatin', 'normal_chromatin', 'Mitosis']];

#Selecting the column representing the target or outcome
y = data[['Class']];

#Split the data 75/25 between training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1);

#Instantiate the linear support vector machine
clf = svm.LinearSVC(C=C_var);

#Use the training data to fit the SVM
clf.fit(x_train, y_train);

#Print results
print("Score: ", clf.score(x_test, y_test));
print("Coefficients: ", clf.coef_);
print("Intercept: ", clf.intercept_);
