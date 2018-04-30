import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)


#Importing DataSet
dataset = pd.read_csv("kc_house_data.csv")
space=dataset['sqft_living']
price=dataset['price']
data = dataset.drop('price', axis =1)
x = np.array(space).reshape(-1, 1)
y = np.array(price)

#Splitting the data into Train and Test
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data, y, test_size=1/3, random_state=0)
print(xtrain)
print("--------------------------------\n", xtest)
print("=================================\n", len(ytrain), len(ytest))

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(ytrain, ytest)

#Predicting the prices
pred = regressor.predict(xtest)
print(pred)

#Visualizing the training Test Results
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Visualizing the Test Results
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()


