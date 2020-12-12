# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint as rdt
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures as PF


print('libraries imported')

##STEP 1 - Import the data used for this project
#Import the solar generation data
Carbon = pd.read_excel('bp-stats-review-2019-all-data.xls', sheet_name = 'Carbon Dioxide Emissions', skiprows = 2)
Carbon.rename(columns={'Million tonnes of carbon dioxide':'Countries'}, inplace =True)
Carbon.set_index('Countries', inplace=True)
# print(Carbon.shape)

#Check the null values to drop any column/row with only NaN values
plt.figure(figsize=(12,8))
sns.heatmap(Carbon.isnull(),cbar=False)
plt.show()

Carbon.dropna(how='all', axis=0, inplace=True)
Carbon.dropna(how='all', axis=1, inplace=True)
Carbon.fillna(0, inplace=True)
Carbon = Carbon.iloc[:Carbon.index.get_loc('Total World'),:Carbon.columns.get_loc(2018)+1]

# print(Carbon.shape)

#Store the index values
places = Carbon.index

#Doublecheck if we took out the columns and rows with NaN values
plt.figure(figsize=(12,8))
sns.heatmap(Carbon.isnull(),cbar=False)
plt.show()

#Choose randomly the Place or Region
number = rdt(0,Carbon.shape[0])

if "Total" in places[number]:
    print('The choosen region is: ',places[number])
else:
    print('The choosen place is: ',places[number])

#Take the data from the table
df = Carbon.loc[places[number]]

#Create a line graph
a=df.transpose()
a.plot(kind='line')
plt.title('Carbon Emission for {}'.format(places[number]))
plt.xlabel('Years')
plt.ylabel('Carbon Emission (Million tonnes)')
plt.show()

#provide data
x = np.array([df.index]).reshape((-1,1))
y = np.array(df.iloc[:])
# print(x)
# print(y)

print('--------')
print('Linear regression')
##STEP 2 - Linear regression
#create and fit the model
model = LR().fit(x,y)

#Get the result score
r_sq = model.score(x,y)
print('The R squared is: ',r_sq)
print(f'The equation is: a*{model.coef_} + {model.intercept_}')


print('--------')
print('Polynomial regression')
##STEP 3 - Polynomial regression
#Transform input data
for i in range(2,5):
    print('The degree of the regression is: ', i)
    X = PF(degree=i, include_bias=False).fit_transform(x)
    
    #Create and fit the model
    model_PR = LR().fit(X,y)
    
    #Get the result score
    r_sq = model_PR.score(X,y)
    print('The R squared is: ',r_sq)
    print('Coefficients: ', model_PR.coef_)
    print('intercept: ',model_PR.intercept_)
    
    print('--')



