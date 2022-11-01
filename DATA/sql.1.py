 
#NumPy module is imported to help us work the arrays that we will be providing in the code,we import it from sklearn which implements regularized logistic regression using the linear model in this case
import numpy
from sklearn import linear_model

#at this point,we store the idependent variables of x and y
#.reshape is used to reshape values of x into a column for the function .LogisticRegression() to work
#values are used as boolean functions even though they cannot be declared boolean in this scenario, 0 for false and 1 for true
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
#method LogisticRegression from sklearn is used to create a logistic regression object
#This regression object has a method called fit() which takes independent and dependent values as parameters and assigns data to the regression object,data that describes relationships.
logr = linear_model.LogisticRegression()
logr.fit(X,y)

#based on the logistic regression object we formed, we can now determine the nature of this object according to its size, 4.21mm in this case.
predicted = logr.predict(numpy.array([4.21]).reshape(-1,1))
print(predicted)
# The prediction output case is above the required value.
#logistic regression in data mining aims to solveclassification problems by predicting categorical outcomes.
