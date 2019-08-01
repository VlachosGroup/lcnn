# -*- coding: utf-8 -*-
# Copyright (c) Vlachos Group, Jung Group
# GNU v3.0 license

__author__ = 'Geun Ho Gu'
__copyright__ = "Copyright 2019, Vlachos Group, Jung Group"
__version__ = "1.0"
__maintainer__ = "Geun Ho Gu"
__email__ = "googhgoo@hotmail.com"
__date__ = "July 31, 2019"

import timeit
import numpy as np
import tensorflow as tf
import os
import json
__all__ = [
        'Hook'
        ]
        
class Hook(object):
    """Record training result during training. 
    """
    def __init__(self,csvlogpath,restart=True,console_print=False,GetBestOnReduce=True):
        """Initialize Hook Class
        
        parameters
        ----------
        csvlogpath : string. path to log
        restart : boolean. whether the training is restarted or not
        console_print : boolean. If true, results are printed in the console directly as well
        GetBestOnReduce : boolean. Whether to restore the best model when learning rate is decayed.
        """
        self.csvlogpath = csvlogpath
        self.restart = restart
        self.console_print = console_print
        self.GetBestOnReduce = GetBestOnReduce       
    def Begin_Train(self,patience,final_lr):
        """This is called when training is started
        
        parameters
        ----------
        patience : integer. Number of iteration before decaying on plateau. see Trainer.
        final_lr : final learning rate over which the training stops. 
        """
        self.final_lr = final_lr
        self.converged = False
        self.patience = patience
        if not self.restart:
            with open(self.csvlogpath,'w') as f:
                log = 'time,learn_rate,TrainMeanAE,TrainRMSE,ValMeanAE,ValRMSE,TestMeanAE,TestRMSE\n'
                f.write(log)
        if self.console_print:
            log = "{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}".format(\
                   'time','learn_rate','TrainMeanAE','TrainRMSE','ValMeanAE','ValRMSE','TestMeanAE','TestRMSE')
            print(log)
        self.best_val_rmse = float('Inf')
        self.start_time = timeit.default_timer()
        self.counter = 0
        
    def Begin_Epoch(self):
        """This is called when epoch starts
        """
        self.Train_res = []
        
    def Begin_TrainBatch(self):
        """This is called when batch training starts
        """
        pass
    
    def End_TrainBatch(self,Y,Yp):
        """This is called when batch training ends
        """
        self.Train_res.append(Y-Yp)
        
    def End_Epoch(self):
        """This is called when epoch is finished
        """
        self.Train_res = np.concatenate(self.Train_res)
            
    def Begin_Val(self):
        """This is called when validation starts
        """
        self.Val_res=[]
        
    def Begin_ValBatch(self):
        """This is called when validation batch starts
        """
        pass
    
    def End_ValBatch(self,Y,Yp):
        """This is called when validation batch ends
        """
        self.Val_res.append(Y-Yp)
        
    def End_Val(self):
        """This is called when validation ends
        """
        self.Val_res = np.concatenate(self.Val_res)
        
    def Begin_Test(self):
        """This is called when test starts
        """
        self.Test_res=[]
        
    def Begin_TestBatch(self):
        """This is called when test batch starts
        """
        pass
    
    def End_TestBatch(self,Y,Yp):
        """This is called when test batch ends
        """
        self.Test_res.append(Y-Yp)
        
    def End_Test(self):
        """This is called when test ends
        """
        if self.Test_res:
            self.Test_res = np.concatenate(self.Test_res)
        
    def End_Train(self,sess,saver,path,global_step,current_lr):
        """This is called when training is finished
        
        parameters
        ----------
        sess : tensorflow session
        saver : tensorflow model savor
        path : path to model 
        global_step : global step tensor for adam algorithm
        current_lr : float. current learning rate. 
        """
        with open(self.csvlogpath,'a') as f:
            log = str(timeit.default_timer() - self.start_time) + ','
            log += str(sess.run(current_lr)) + ','
            log += str(np.mean(np.abs(self.Train_res))) + ','
            log += str(np.sqrt(np.mean(self.Train_res**2))) + ','
            log += str(np.mean(np.abs(self.Val_res))) + ','
            val_rmse = np.sqrt(np.mean(self.Val_res**2))
            log += str(val_rmse) + ','
            if self.Test_res !=[]:
                log += str(np.mean(np.abs(self.Test_res))) + ','
                log += str(np.sqrt(np.mean(self.Test_res**2))) + ','
            else: 
                log += 'none,'
                log += 'none,'
            f.write(log+'\n')
        if self.console_print:
            if self.TestRes!=[]:
                log = "{:12.1f}{:12.1e}{:12.4f}{:12.4f}{:12.4f}{:12.4f}{:12.4f}{:12.4f}".format(\
                       timeit.default_timer() - self.start_time,sess.run(current_lr), 
                       np.mean(np.abs(self.Train_res)),\
                       np.sqrt(np.mean(self.Train_res**2)),np.mean(np.abs(self.Val_res)),\
                       val_rmse,np.mean(np.abs(self.Test_res)),np.sqrt(np.mean(self.Test_res**2)))
            else:
                log = "{:12.1f}{:12.1e}{:12.4f}{:12.4f}{:12.4f}{:12.4f}".format(\
                       timeit.default_timer() - self.start_time,sess.run(current_lr), 
                       np.mean(np.abs(self.Train_res)),\
                       np.sqrt(np.mean(self.Train_res**2)),np.mean(np.abs(self.Val_res)),\
                       val_rmse)
            print(log)
        if val_rmse < self.best_val_rmse:
            self.best_val_rmse = val_rmse
            self.counter = 0
            saver.save(sess,os.path.join(path,"bestmodel.ckpt"))
            if isinstance(self.Test_res,list):
                test_res = self.Test_res
            else:
                test_res = self.Test_res.tolist()
            json.dump({'train':self.Train_res.tolist(),'validation':self.Val_res.tolist(),'test':test_res},open(os.path.join(path,'Residual.json'),'w'))
        else:
            self.counter += 1
        # decay
        if self.counter == self.patience:
            if self.GetBestOnReduce:
                saver.restore(sess,os.path.join(path,"bestmodel.ckpt"))
            sess.run(tf.assign_add(global_step,1))
            self.counter = 0
            if sess.run(current_lr)<self.final_lr:
                self.converged = True
