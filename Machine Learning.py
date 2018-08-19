# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:00:18 2017

@author: ram
"""

import numpy as np





from sklearn.cross_validation import train_test_split

values=np.arange(5)

var=np.sum(((values-np.mean(values))**2)/len(values))

var


def mean( values):
        # compute the mean of a list of numbers
        # Todo: Calculate the mean of a list of numbers using numpy
        ## Input: A numpy array with 1 dimension
        ## return: the mean value
        return np.mean(values)

mean(values)

def variance(values):
        # compute the variance in a list of n numbers
        # Todo: Calculate the sample variance of a list of numbers using numpy
        ## Input: A numpy arraty with 1 dimension 
        var=np.sum(((values-mean(values))**2)/len(values))
        ## Return: the sample variance multiplied by the number of samples 
        return var

variance(values)

    def covariance( x, y):
        # Compute covariance between x and y
        ## Todo: Calculate the covariance
        ## Inputs: X and y, the input and the predictor
        ## Return: the covariance of x and y
        cov=np.sum(((x-mean(x))*(y-mean(y)))/len(x))
        ### Reference: https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line
        return cov
covariance(values,np.random.random(5))
def fit( x, y):
        # Fit the regression model by computing coefficients
        # Todo: Compute the mean, variance and covariance
        # use the above three values and the dataset to compute the parameters
        ## Inputs: the predictor x and the target y
        ## Returns: N/A
        var_x=variance(x)
        cov_x_y=covariance(x,y)
        theta_1 = cov_x_y/var_x
        theta_0 = mean(y)-theta_1*mean(x)
        coeff_ = [theta_1, theta_0]
        return coeff_

a=fit(values,np.random.random(5))
a

def predict(x):
        # make predictions on new data
        # use the parameters and the linear equation to compute the output
        ## Inputs: sample data to predict using
        ## Returns: Target predictions
        pred_x=a[0]+a[1]*x
        return pred_x

predict(np.random.random(3))

predict(values)

def score(x, y):
        # Compute the r_2 score of the model
        #r_2=sum((y-predict(x))**2)/sum((y-mean(y))**2)
        # use the predict method to get the predictions
        pred_x=predict(x)
        # Calculate the SSE for the model
        sse=sum((y-predict(x))**2)
        # Calculate the SSE for a model that predicts the y_mean
        sst=sum((y-mean(y))**2)
        # Compute the r2 score
        r_2=1-sse/sst
        ## Inputs: the predictor x and the target y
        ## Returns: The r2 score of the model
        return r_2

score(values,np.random.random(5))