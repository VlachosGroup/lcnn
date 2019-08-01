# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 11:05:40 2018

@author: user
"""
import subprocess
from itertools import product
import os
import json
import csv
import numpy as np
import timeit
from scipy.stats import sem
import sys
import random

class Sampler(object):
    def __init__(self,L2NormRange,NConvRange,Radius,NFeatureRange,NFinalFeatureRange):
        self.L2NormRange = L2NormRange
        self.NConvRange = NConvRange
        self.Radius = Radius
        self.NFeatureRange = NFeatureRange
        self.NFinalFeatureRange = NFinalFeatureRange
    def Sample(self):
        Lambda = "{:.3e}".format(10**np.random.uniform(self.L2NormRange[0],self.L2NormRange[1]))
        NConv = str(np.random.randint(self.NConvRange[0],self.NConvRange[1]+1))
        Radius = str(random.sample(self.Radius,1)[0])
        NFeature = str(np.random.randint(self.NFeatureRange[0],self.NFeatureRange[1]))
        NFFeature = str(np.random.randint(self.NFinalFeatureRange[0],self.NFinalFeatureRange[1]))
        return Lambda, NConv, Radius, NFeature, NFFeature

if __name__ == '__main__':
    """Perform Hyperparamter Optimization.
    
    This is one way to do hyperparameter optimization. kFold test is used with
    holdout cross validation. random hyperparameter search is used.
    
    parameters
    ----------
    logpath : path that will record hyperparmater opimization result
    nTest : Test fold
    nsample : number of random hyperparameter sampling
    L2Penalty : range for L2 norm penalty
    nconv : range for number of convolution
    radius : list off cutoff radius used
    feature : range for number of features
    final_feature : range for number of sitewise feature
    modelspath : path to where the model will be
    ndata : total number of data
    
    return
    ------
    check log file for the hyperparameter optimization result. 
    """
    ############################ User Input ###################################
    logpath = 'log'
    nTest = 10
    nsample = 2
    L2Penalty=[-3,-5]
    nconv=[1,2]
    radius = [3.00,6.00] # List of radius, not range
    feature = [2,20]
    final_feature=[2,4]
    modelspath = './models'
    ndata = 648
    ###########################################################################
    # Generate Split
    ## 72 : 18 : 10, train, validation, test
    testfolds = np.array_split(np.random.permutation(648),10)
    Splits = []
    for i in range(nTest):
        tv = np.concatenate([testfolds[j] for j in range(nTest) if j != i])
        T = testfolds[i]
        tv = np.random.permutation(tv)
        t = tv[:int(np.around(len(tv)*0.8))]
        v = tv[int(np.around(len(tv)*0.8)):]
        Splits.append({'Test':T.tolist(),'Validation':v.tolist(),'Train':t.tolist()})
    json.dump(Splits,open('Splits.json','w'))
        
    ParamSampler = Sampler(L2Penalty,nconv,radius,feature,final_feature)
    
    if not os.path.isdir(modelspath):
        os.mkdir(modelspath)
        
    log = "{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>12s}{:>12s}{:>12s}".format(\
                   'time','set','feature','Ffeature','nconv','radius','Lambda','BestValRMSE','TrainRMSE','TestRMSE')
        
    
    n = 0 
    with open(logpath,'w') as f:
        f.write(log+'\n')
    for _ in range(0,nsample):
        Lambda, NConv, Radius, NFeature, NFFeature = ParamSampler.Sample()
        min_vals = []
        min_trains = []
        min_tests = []
        for j in range(nTest):
            start_time = timeit.default_timer()
            # make command
            modelpath = os.path.join(modelspath,'model_' + NFeature + '_' + NFFeature + '_' + NConv + '_' + Radius + '_' + Lambda + '_' + str(j))
            if not os.path.isdir(modelpath):
                os.mkdir(modelpath)
            # dump split file
            json.dump(Splits[j],open(os.path.join(modelpath,'split.json'),'w'))
            
            cmd = 'python ../../bin/lcnn_run.py train '+modelpath+' ./data ' + \
                '--L2Penalty '+Lambda+' --nconv '+NConv+' --cutoff '+Radius+' --feature '+NFeature+\
                ' --final_feature ' + NFFeature
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate() 
            
            # get result
            
            with open(os.path.join(modelpath,'train_log.csv')) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                val_rmse = []
                train_rmse = []
                test_rmse = []
                line_count = 0
                for row in csv_reader:
                    if line_count != 0:
                        train_rmse.append(float(row[3]))
                        val_rmse.append(float(row[5]))
                        test_rmse.append(float(row[7]))
                    line_count += 1
                i = np.argmin(val_rmse)
                min_vals.append(val_rmse[i]*1000)
                min_trains.append(train_rmse[i]*1000)
                min_tests.append(test_rmse[i]*1000)
            log = "{:10.1f}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>12.6f}{:12.6f}{:12.6f}".format(\
               timeit.default_timer() - start_time, str(j),NFeature, 
               NFFeature,NConv,Radius,Lambda,\
               min_vals[-1],min_trains[-1],min_tests[-1])
            with open(logpath,'a') as f:
                f.write(log+'\n')
        log = "{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>8.1f}±{:>3.1f}{:>8.1f}±{:>3.1f}{:>8.1f}±{:>3.1f}".format(\
           '','Average',NFeature,NFFeature,NConv,Radius,Lambda,\
           np.mean(min_vals),sem(min_vals),np.mean(min_trains),sem(min_vals),np.mean(min_tests),sem(min_vals))
        with open(logpath,'a') as f:
            f.write(log+'\n')

