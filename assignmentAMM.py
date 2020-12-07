#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:37:51 2020
@author: woutkonings
"""

import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
import csv



def logit_function(alpha, beta, price):
    """
    Function to calculate a univariate logit function
    """
    
    return math.exp(alpha + beta * price ) / ( 1 + math.exp(alpha + beta * price))


def likelihood_single_obs(alpha, beta, price, y):
    """
    Function to calculate the likelihood for a binary variable with a logit probability
    """
    
    return math.pow(logit_function(alpha, beta, price), y) * math.pow(1 - logit_function(alpha, beta, price), 1 - y)


def LogL(theta, segment_prob, y, X):
    """
    Function to calculate the Log Likelihood for logistic binary panel data
    """
    
    start_sum = 0
    for i in range(y.shape[0]):
        for t in range(y.shape[1]):
            sum_total = 0
            for j in range(int(theta.shape[0]/2)):
                to_add = segment_prob[j] * likelihood_single_obs(theta[2 * j], theta[2 * j + 1], X[t, 1], y[i, t])
                sum_total = sum_total + to_add
            log_sum_total = math.log(sum_total)
            start_sum = start_sum + log_sum_total
    return start_sum

def likelihood_single_individual(alpha, beta, X, y_individual):
    """
    Function to calculate the likelihood for a single individual in a panel data set
    """
    
    total_product = 1
    for t in range(y_individual.shape[0]):
        total_product = total_product * likelihood_single_obs(alpha, beta, X[t,1], y_individual[t])
    return total_product


def Estep(theta, pi, y, X):
    """
    Function for the expectation step in the EM algorithm
    
    Parameters
    ----------
    theta: array
        Array of the alpha's and beta's used in the mixture logit model. Input in the form [alpha_1, beta_1, ...., alpha_k, beta_k]
    pi: array
        Array of the segment probabilities
    y: Array
        N-by-T binary array of the dependent variable
    X: Array
        T-by-2 array with a constant and the prices over time
        
    Returns
    --------
    P: array
        N-by-T array of of individual conditional cluster probabilities
    """
    
    P = np.zeros((250, k))
    for i in range(250):
        denom = 0
        for j in range(k):
            denom += pi[j] * likelihood_single_individual(theta[2 * j], theta[2 * j + 1], X, y[i,:])
        for j in range(k):
            P[i,j] = pi[j] * likelihood_single_individual(theta[2 * j], theta[2 * j + 1], X, y[i,:]) / denom
    return P


def Mstep(W, y, X, starting_theta, starting_pi):
    """
    Function for the maximization step in the EM algorithm
    
    
    Parameters
    ----------
    W: array
        N-by-T array of of individual conditional cluster probabilities
    y: Array
        N-by-T binary array of the dependent variable
    X: Array
        T-by-2 array with a constant and the prices over time
    starting_theta: array
        Array of the alpha's and beta's used in the mixture logit model. Input in the form [alpha_1, beta_1, ...., alpha_k, beta_k]
    starting_pi: array
        Array of the segment probabilities
        
    Returns
    --------
    new_theta: array
        new theta values
    new_pi: array
        new pi values
    """
    
    denom = W.sum(axis = 0)
    new_pi = denom / denom.sum()

    res = minimize(function_mstep, x0 = starting_theta, args =(W,y,X),  method='Nelder-Mead', options = {'maxiter': 250, 'disp': True} )
    new_theta = res.x
    return new_theta, new_pi

def function_mstep(starting_theta, W, y, X):
    """
    Log likelihood function to be optimized in the Mstep
    
    Parameters
    ----------
    starting_theta : array
        Array of the alpha's and beta's used in the mixture logit model. Input in the form [alpha_1, beta_1, ...., alpha_k, beta_k]
    W : TYPE
        N-by-T array of of individual conditional cluster probabilities
    y : TYPE
        N-by-T binary array of the dependent variable
    X : TYPE
        T-by-2 array with a constant and the prices over time

    Returns
    -------
    float
        The log likelihood objective value to be optimized

    """
    start_sum = 0
    for j in range(W.shape[1]):
        for i in range(W.shape[0]):
            inner_sum = 0
            for t in range(y.shape[1]):
                a = likelihood_single_obs(starting_theta[2 * j], starting_theta[2 * j + 1], X[t,1], y[i, t])
                if a == 0:
                    # underflow error
                    a = sys.float_info.min
                inner_sum = inner_sum + math.log(a)
            start_sum = start_sum + W[i,j] * inner_sum
    return -1 * start_sum


def EM(y, X, k):
    """
    Function that initialises random starting values for the logit parameters and the segment probabilities 
    and then iterates the E- and M-step until a stopping criterium is met
    
    
    Parameters
    ----------
    y : TYPE
        N-by-T binary array of the dependent variable
    X : TYPE
        T-by-2 array with a constant and the prices over time
    k : int
        nubmer of segments

    Returns
    -------
    theta: array
        Array of the estimated alpha's and beta's
    pi: array
        Array of the estimated segment probabilities

    """
    
    print("Running EM for k = " + str(k))
    
    #Set stopping criteria. Either maximum number of iterations or segment probabilities don't improve anymore.
    epsilon = 0.001
    count_iter = 0
    max_iter = 50
    
    #Lists to store results
    logLs = []
    pis = []
    
    #Set random starting values
    theta = np.random.uniform(low = -0.5, high = 0.5, size = 2 * k)
    pi = np.full(k, 1/k)

    stopping_condition = False

    while not stopping_condition and count_iter < max_iter:
        
        print("\n")
        print("at iteration " + str(count_iter + 1) + " of " + str(max_iter) + " of the EM function.")
        P = Estep(theta, pi, y, X)
        new_theta, new_pi = Mstep(P, y, X, theta, pi)
        distance_pi = np.linalg.norm(new_pi - pi)
        if distance_pi < epsilon :
            print("stopping as segment probabilities are not imporving anymore")
            stopping_condition = True
        theta = new_theta
        pi = new_pi
        count_iter = count_iter + 1
        np.set_printoptions(precision=3)
        
        #append results to list
        logLs.append(LogL(theta, pi, y, X))
        pis.append(pi)
        
        #print(LogL(theta, pi, y, X))
        #print(pi)
        #print(theta)
    return logLs, theta, pis


def Estimate(y, X, k):
    """
    runs the EM function a number of times with different (random) starting values 
    in order to prevent getting stuck at a local optimum.
    
    Parameters & returns same as for EM
    ----------
    """
    
    print("running Estimate function")
    results = dict()
    for i in range(10):
        print("\n")
        print("NOW in LOOP " + str(i + 1) + " of the estimate loop")
        logLs, theta, pis = EM(y, X, k)
        
        #make tuple for theta and the last value of the pi result list
        params = (theta, pis[-1])
        print("PARAMS")
        print(params)
        results[logLs[-1]] = params
    return results

def transform_pi_to_gamma(pi):
    k = len(pi)
    gamma = np.zeros(k - 1)
    for i in range((k-1)):
        gamma[i] = math.log(pi[i]) - math.log(pi[k-1])
    
    return gamma

def transform_gamma_to_pi(gamma):
    k_minus_one = len(gamma)
    k = k_minus_one + 1
    pi = np.zeros(k)
    gamma_sum = sum(np.exp(gamma))
    for i in range(k):
        print(i)
        if i == (k - 1):
            pi[i] = 1 / (1 + gamma_sum)
        else:
            pi[i] = np.exp(gamma[i]) / (1 + gamma_sum)
    
    return pi

def new_logL(theta, gamma, y, X):
    
    pi = transform_gamma_to_pi(gamma)
    
    value = LogL(theta, pi, y, X)
    return value

if __name__ == "__main__":
    data = pd.read_csv('433246.csv')
    Y_df  = pd.DataFrame(index=range(250),columns=range(25))
    X_df  = pd.DataFrame(index=range(25),columns=range(2))


    for index, row in data.iterrows():
        individual_index = int(np.mod(row['Person'],250))
        week = int(row['Week'])
        Y_df.iat[individual_index, week - 1] = row['y']
        X_df.iat[week - 1, 0] = 1
        X_df.iat[week - 1, 1] = row["price"]
    
    print("data loaded")

    k = 3
    Y = Y_df.to_numpy()
    X = X_df.to_numpy()
    
    results = Estimate(Y, X, k)
    
    with open('data.csv', 'w') as f:
        for key in results.keys():
            f.write("%s, %s\n" % (key, results[key]))
    
    best = max(results.keys())
    bestParams = results[best]
    bestTheta = bestParams[0]
    bestPi = bestParams[1]
    bestGamma = transform_pi_to_gamma(bestPi)
    
    hessian_ = hessian(new_logL)
    H = hessian_(bestTheta, bestGamma, Y, X)
    cov = np.inv(-1 * H)
    
    #Test the performance of EM step and plot the loglikelihoods and the segment probabilities of the EM loop
    
    logLs, theta, pis = EM(Y, X, k)
    plt.plot(logLs)
    pis_array = np.asarray(pis)
    fig, ax = plt.subplots()
    for i in range(k):
        ax.plot(pis_array[:,i])
    