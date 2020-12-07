#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sun Dec  6 18:37:51 2020

@author: woutkonings
"""

import pandas as pd
import numpy as np
import math


def LogL(theta, segment_prob, y, X):
    start_sum = 0
    for i in range(y.shape[0]):
        for t in range(y.shape[1]):
            sum_total = 0
            for j in range(int(theta.shape[0]/2)):
                to_add = segment_prob[j] * likelihood_single_obs(theta[2 * j], theta[2 * j + 1], X[t, 1], y[i, t])
                sum_total = sum_total + to_add
                print(sum_total)
            log_sum_total = math.log(sum_total)
            start_sum = start_sum + log_sum_total
    print(start_sum)
    return start_sum


    
def logit_function(alpha, beta, price):
    return math.exp(alpha + beta * price ) / ( 1 + math.exp(alpha + beta * price))


def likelihood_single_obs(alpha, beta, price, y):
    return math.pow(logit_function(alpha, beta, price), y) * math.pow(1 - logit_function(alpha, beta, price), 1 - y)


def likelihood_single_individual(alpha, beta, X, y_individual):
    total_product = 1
    for t in range(y_individual.shape[0]):
        total_product = total_product * likelihood_single_obs(alpha, beta, X[t,1], y_individual[t])
    return total_product

def Estep(Theta, Pi, y, X):
    P = np.zeros((250, k))
    for i in range(250):
        denom = 0
        for j in range(k):
            denom += Pi[j] * likelihood_single_individual(Theta[j], Theta[j + 1], X, y[i,:])
        for j in range(k):
            P[i,j] = Pi[j] * likelihood_single_individual(Theta[j], Theta[j + 1], X, y[i,:]) / denom
    return P

"""
def Mstep(W, y, X, starting_theta, starting_pi):
    start_sum = 0
    for j in range(W.shape[1]):
        for i in range(W.shape[0]):
            inner_sum = 0
            for t in range(y.shape[1]):
                likelihood_single_obs(alpha, beta, X[t,1], y_individual[t])
            
"""

data = pd.read_csv('433246.csv')

Y_df  = pd.DataFrame(index=range(250),columns=range(25))
X_df  = pd.DataFrame(index=range(25),columns=range(2))


for index, row in data.iterrows():
    individual_index = int(np.mod(row['Person'],250))
    week = int(row['Week'])
    Y_df.iat[individual_index, week - 1] = row['y']
    X_df.iat[week - 1, 0] = 1
    X_df.iat[week - 1, 1] = row["price"]


k = 4
Y = Y_df.to_numpy()
X = X_df.to_numpy()
Theta = np.ones(k*2)
Pi = np.full(k, 1/k)
Theta[3] = 0.96


LogL(Theta, Pi, Y, X)
P = Estep(Theta, Pi, Y, X)
np.sum(P, axis = 1)


def EM(y, X, k):

    epsilon = 0.01
    old_logL = 0
    theta = np.ones(k*2)
    pi = np.full(k, 1/k)
    
    P = Estep(Theta, Pi, y, X)
    new_values = Mstep(P, y, X, Theta, Pi)
    theta = new_values[0]
    pi = new_values[1]
    
    LogL = 

New_Pi = np.zeros(k)
denom = P.sum(axis = 0)
New_Pi = denom / denom.sum()

def Estimate(y, X, k):
    
    results = new_dict()
    
    for i in range(10):
        thetha, pi = EM(y, X, k)
        params = (theta, pi)
        results.append(params, LogL(params[0], params[1], y, X))
    
    sorted_results = sorted(results)
    return list(sorted_results.keys())[0]


