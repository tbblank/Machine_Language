import pandas as pd;
import seaborn as sea;
from sklearn.cross_validation import train_test_split;
from sklearn.linear_model import LinearRegression;

data = pd.read_csv('Z:/School_Training/Machine Language/assigns/AP1/auto-mpg.csv');
X = data[['Displacement', 'Horsepower', 'Weight', 'Acceleration']];
#X = data[['Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Cylinders']];
print(X.head());
#print(type(X));

y = data[['MPG']];
#print(y.head());
#print(type(y));

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1);

linreg = LinearRegression();
linreg.fit(x_train, y_train);

#=======================================
#Overall regression

print("Overall linear regression intercept: ", linreg.intercept_);
print("Overall linear regression coefficient: ", linreg.coef_);
print("Overall R^2 Score: ", linreg.score(X, y));

#=======================================
#Regressing for displacement
disp = X[['Displacement']];
xd_train, xd_test, yd_train, yd_test = train_test_split(disp, y, random_state=1);
linreg_disp = LinearRegression();
linreg_disp.fit(xd_train, yd_train);
#print(linreg_disp.coef_);
print("Displacement R^2 Score: ", linreg_disp.score(disp, y));

#=======================================
#Regressing for horsepower
hors = X[['Horsepower']];
xh_train, xh_test, yh_train, yh_test = train_test_split(hors, y, random_state=1);
linreg_hors = LinearRegression();
linreg_hors.fit(xh_train, yh_train);
#print(linreg_hors.coef_);
print("Horsepower R^2 Score: ", linreg_hors.score(hors, y));

#=======================================
#Regressing for weight
wei = X[['Weight']];
xw_train, xw_test, yw_train, yw_test = train_test_split(wei, y, random_state=1);
linreg_wei = LinearRegression();
linreg_wei.fit(xw_train, yw_train);
#print("Weight coefficient: ", linreg_wei.coef_);
print("Weight R^2 Score: ", linreg_wei.score(wei, y));

#=======================================
#Regressing for acceleration
acc = X[['Acceleration']];
xa_train, xa_test, ya_train, ya_test = train_test_split(acc, y, random_state=1);
linreg_acc = LinearRegression();
linreg_acc.fit(xa_train, ya_train);
#print(linreg_acc.coef_);
print("Acceleration  R^2 Score: ", linreg_acc.score(acc, y));

#=======================================
#Regressing for cylinders
#cyl = X[['Cylinders']];
#xc_train, xc_test, yc_train, yc_test = train_test_split(cyl, y, random_state=1);
#linreg_cyl = LinearRegression();
#linreg_cyl.fit(xc_train, yc_train);
#print(linreg_disp.coef_);
#print("Cylinders R^2 Score: ", linreg_cyl.score(cyl, y));