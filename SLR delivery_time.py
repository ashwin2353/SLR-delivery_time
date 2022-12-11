# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 12:17:02 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("delivery_time.csv")
df.shape
df.head()
df.dtypes

df["Delivery Time"].value_counts()
df["Sorting Time"].value_counts()

# blanks
df.isnull().sum()

# finding out the duplicated rows
df.duplicated()
df[df.duplicated()]
# there are no duplicated rows

# finding out the duplicated columns
df.columns.duplicated()
# therefore no duplicated columns

#========================================================================
# Spliting the variables as X and Y
Y = df["Sorting Time"]
X = df[["Delivery Time"]]

#========================================================================
# Histogram
df["Delivery Time"].hist()
df["Delivery Time"].skew()
# Therefore Delivery time shows that it is -ve skewness 
df["Sorting Time"].hist()
df["Sorting Time"].skew()
# Therefore sorting time shows that is also -ve skewness 
#========================================================================= 
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df["Delivery Time"])
plt.title("Delivery Time PDF")

plt.subplot(122)
stats.probplot(df["Delivery Time"],dist="norm",plot=plt)
plt.title("delivery time QQ plot")
plt.show()
# by this i have seen the probability distribusion function and QQ plot of Delivery time
#==========================================================================
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df["Sorting Time"])
plt.title("Sorting Time") 

plt.subplot(122)
stats.probplot(df["Sorting Time"],dist="norm",plot=plt)
plt.title("Sorting Time QQ plot")
plt.show()
# by this i have seen the probability distribusion function and QQ plot of Sorting time
#=========================================================================
# scatter plot
import matplotlib.pyplot as plt
plt.scatter(X,Y,color="black")
plt.show
# in scatter plot we get to know that X and Y variabels are in +ve relationship
#=========================================================================
# Boxplot
df.boxplot(column="Delivery Time",vert=False)
import numpy as np
Q1 = np.percentile(df["Delivery Time"],25)
Q2 = np.percentile(df["Delivery Time"],50)
Q3 = np.percentile(df["Delivery Time"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Delivery Time"]<LW) | (df["Delivery Time"]>UW)]
len(df[(df["Delivery Time"]<LW) | (df["Delivery Time"]>UW)])
# Therefore in Delivery time variabel there are Zero outlaiers 

#========================================================================
df.boxplot(column="Sorting Time",vert=False)
import numpy as np
Q1 = np.percentile(df["Sorting Time"],25)
Q2 = np.percentile(df["Sorting Time"],50)
Q3 = np.percentile(df["Sorting Time"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Sorting Time"]<LW) | (df["Sorting Time"]>UW)]
len(df[(df["Sorting Time"]<LW) | (df["Sorting Time"]>UW)])
# Therefore in Sorting time variabel there are Zero outlaiers 

#=======================================================================
# Model fitting 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

# B0 
LR.intercept_

# B1
LR.coef_

Y_pred = LR.predict(X)
Y_pred

#scatter plot
import matplotlib.pyplot as plt
plt.scatter(X,Y,color="black",alpha=0.5)
plt.scatter(X,Y_pred,color='green',alpha=0.5)
#========================================================================
# matrix

# MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y, Y_pred)
print('Mean squared error',mse.round(3))

# RMSE
import numpy as np
print("Root mean squared error",np.sqrt(mse).round(3))

#===========================================================================
# log Transformation
X_log= np.log(X)

Y_pred_Transform = LR.predict(X_log)
Y_pred_Transform

# matrix
# MSE
from sklearn.metrics import mean_squared_error
mse_1 = mean_squared_error(Y, Y_pred_Transform)
print('Mean squared error',mse.round(3))

# RMSE
import numpy as np
print("Root mean squared error",np.sqrt(mse_1).round(3))
#===========================================================================
# square Root Transformation
X_log1 = np.sqrt(X)
 
Y_pred_Transform = LR.predict(X_log1)
Y_pred_Transform

# matrix
# MSE
from sklearn.metrics import mean_squared_error
mse_2 = mean_squared_error(Y, Y_pred_Transform)
print('Mean squared error',mse.round(3))

# RMSE
import numpy as np
print("Root mean squared error",np.sqrt(mse_2).round(3))
#=========================================================================
# cube Root Transformation 
X_log2 = np.cbrt(X)

Y_pred_Transform = LR.predict(X_log2)
Y_pred_Transform

# matrix

# MSE
from sklearn.metrics import mean_squared_error
mse_3 = mean_squared_error(Y, Y_pred_Transform)
print('Mean squared error',mse.round(3))
# RMSE
import numpy as np
print("Root mean squared error",np.sqrt(mse_3).round(3))
#=============================================================================
import statsmodels.api as sma
X_new = sma.add_constant(X)
X_new
lm2 = sma.OLS(Y,X_new).fit()
lm2.summary()
#=======================================================================
# VIF
RSS = np.sum(Y_pred-Y)**2
Y_mean = np.mean(Y)
TSS = np.sum(Y-Y_mean)**2
R2 = 1-(RSS/TSS)
print("R2:",R2)

vif = 1/(1-R2)
print("VIF value:",vif)
# our model is getting VIF value is 0.028 that is less then 5 so our model is excellent

# Therefore by comparing all the Transformations I find out that without using any Transformation we are giving the best resluts








