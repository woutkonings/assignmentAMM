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


def LogL(theta, segment_prob, y, X):
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
            denom += Pi[j] * likelihood_single_individual(Theta[2 * j], Theta[2 * j + 1], X, y[i,:])
        for j in range(k):
            P[i,j] = Pi[j] * likelihood_single_individual(Theta[2 * j], Theta[2 * j + 1], X, y[i,:]) / denom
    return P


def Mstep(W, y, X, starting_theta, starting_pi):
    denom = W.sum(axis = 0)
    new_pi = denom / denom.sum()
    
    res = minimize(function_mstep, x0 = starting_theta, args =(W,y,X),  method='Nelder-Mead', options = {'maxiter': 100, 'disp': True} )
    new_theta = res.x
    return new_theta, new_pi
    
def function_mstep(starting_theta, W, y, X):
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

    epsilon = 0.01
    old_logL = 0
    theta = np.random.uniform(low = -0.5, high = 0.5, size = 2 * k)
    pi = np.full(k, 1/k)
    
    stopping_condition = False
    count_iter = 0
    max_iter = 10
    while not stopping_condition and count_iter < max_iter:
        print(count_iter)
        P = Estep(theta, pi, y, X)
        new_theta, new_pi = Mstep(P, y, X, theta, pi)
        distance_pi = np.linalg.norm(new_pi - pi)
        if distance_pi < 0.00001 :
            stopping_condition = True
        theta = new_theta
        pi = new_pi
        count_iter = count_iter + 1
        print(LogL(theta, pi, y, X))
        print(pi)
        print(theta)
    return theta, pi
    

def Estimate(y, X, k):
    results = dict()
    for i in range(10):
        thetha, pi = EM(y, X, k)
        params = (theta, pi)
        results[params] = LogL(params[0], params[1], y, X)
    sorted_results = sorted(results)
    return list(sorted_results.keys())[0]


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
    
    
    k = 4
    Y = Y_df.to_numpy()
    X = X_df.to_numpy()
    Thetha = np.ones(k*2)
    Pi = np.full(k, 1/k)
    Thetha[3] = -5
    
    LogL(Thetha, Pi, Y, X)
    P = Estep(Thetha, Pi, Y, X)
    np.sum(P, axis = 1)
    res = Mstep(P, Y, X, Thetha, Pi)
    theta, pi = EM(Y, X, k)