# -*- coding: utf-8 -*-
# Copyright (c) Vlachos Group, Jung Group
# GNU v3.0 license

__author__ = 'Geun Ho Gu'
__copyright__ = "Copyright 2019, Vlachos Group, Jung Group"
__version__ = "1.0"
__maintainer__ = "Geun Ho Gu"
__email__ = "googhgoo@hotmail.com"
__date__ = "July 31, 2019"

import tensorflow as tf
import os
import copy
__all__ = [
        'Trainer'
        ]
        
class Trainer(object):
    def __init__(self,modelpath,model,dataloader,hook,Lambda,dropout_rate,init_lr=1e-2,final_lr=1e-4,decay_rate=0.96,patience=25,n_max_epoch=10000,restart=False):
        """Class for training the model. 
        
        parameters
        ----------
        modelpath : string. path to where the model is
        model : tensorflow based LCNN model.
        dataloader : DataLoader class
        hook : Hook class in Hook.py for recording training results.
        Lambda : float. L2 norm penalty value
        dropout_rate : float. dropout rate
        init_lr : float. initial learning rate
        final_lr : float. final learning rate over which the trainer stops
        decay_rate : float. ratio by which the learning rate decreases.
        patience : integer. number of iteration before decay rate when
            validation score does not improve
        n_max_epoch : integer. maximum number of epoch for each learning rate
        restart : boolean. Whether to restart or not. 
        
        """
        self.modelpath = modelpath
        self.model = model
        self.restart = restart
        self.train_XY = dataloader.GetTrainSet()
        self.validation_XY = dataloader.GetValidationSet()
        self.test_XY = dataloader.GetTestSet()
        self.hook = hook
        self.Lambda = Lambda
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.decay_rate = decay_rate
        self.n_max_epoch = n_max_epoch
        
    def Train(self):
        """Begin Training. 
        """ 
        with tf.Session() as sess:
            saver = tf.train.Saver()
            if self.restart and os.path.exists(os.path.join(self.modelpath,'bestmodel.ckpt')):
                saver.restore(sess, os.path.join(self.modelpath,'bestmodel.ckpt'))
            if not os.path.isdir(self.modelpath):
                os.mkdir(self.modelpath)
                
            #exponential decay
            global_step = tf.Variable(0,dtype=tf.int32,trainable=False)
            lr= tf.train.exponential_decay(
                self.init_lr,
                global_step,
                1,
                self.decay_rate,
                staircase=True,
                )
            #Optimizer
            Optimizer = tf.train.AdamOptimizer(lr)
            Train = Optimizer.minimize(self.model.Loss)
            
            sess.run(tf.global_variables_initializer())
            
            # standardize
            X_mean_T,X_std_T,Y_mean_T,Y_std_T = self.model.GetStatistics()
            X_mean = sess.run(X_mean_T)
            X_std = sess.run(X_std_T)
            Y_mean = sess.run(Y_mean_T)
            Y_std = sess.run(Y_std_T)
            
            train_XY = copy.deepcopy(self.train_XY)
            validation_XY = copy.deepcopy(self.validation_XY)
            test_XY = copy.deepcopy(self.test_XY)
            for batch in train_XY:
                batch['Y'] = (batch['Y']-Y_mean)/Y_std
                batch['X_Sites'] = (batch['X_Sites']-X_mean)/X_std
            for batch in validation_XY:
                batch['Y'] = (batch['Y']-Y_mean)/Y_std
                batch['X_Sites'] = (batch['X_Sites']-X_mean)/X_std
            for batch in test_XY:
                batch['Y'] = (batch['Y']-Y_mean)/Y_std
                batch['X_Sites'] = (batch['X_Sites']-X_mean)/X_std
            
            # begin training
            self.hook.Begin_Train(self.patience,self.final_lr)
            for i in range(0,self.n_max_epoch):
                self.hook.Begin_Epoch()
                for batch in train_XY:
                    self.hook.Begin_TrainBatch()
                    _, _, Yp = sess.run((Train, self.model.Loss, self.model.Y_conf_wise),
                        feed_dict={
                        self.model.Y:batch['Y'],
                        self.model.N_Sites:batch['N_Sites'],
                        self.model.N_Sites_per_config:batch['N_Sites_per_config'],
                        self.model.Idx_config:batch['Idx_Config'],
                        self.model.X_Sites:batch['X_Sites'],
                        self.model.X_NSs:batch['X_NSs'],
                        self.model.Lambda:self.Lambda,
                        self.model.Dropout_rate:self.dropout_rate,
                        self.model.is_training:True})
                    self.hook.End_TrainBatch(batch['Y']*Y_std+Y_mean,Yp*Y_std+Y_mean)
                self.hook.End_Epoch()
                self.hook.Begin_Val()
                for batch in validation_XY:
                    self.hook.Begin_ValBatch()
                    Ypv = sess.run(self.model.Y_conf_wise,
                        feed_dict={
                        self.model.Y:batch['Y'],
                        self.model.N_Sites:batch['N_Sites'],
                        self.model.N_Sites_per_config:batch['N_Sites_per_config'],
                        self.model.Idx_config:batch['Idx_Config'],
                        self.model.X_Sites:batch['X_Sites'],
                        self.model.X_NSs:batch['X_NSs'],
                        self.model.Dropout_rate:1.0,
                        self.model.is_training:False})
                    self.hook.End_ValBatch(batch['Y']*Y_std+Y_mean,Ypv*Y_std+Y_mean)
                self.hook.End_Val()
                self.hook.Begin_Test()
                for batch in test_XY:
                    self.hook.Begin_TestBatch()
                    Ypt = sess.run(self.model.Y_conf_wise,
                        feed_dict={
                        self.model.Y:batch['Y'],
                        self.model.N_Sites:batch['N_Sites'],
                        self.model.N_Sites_per_config:batch['N_Sites_per_config'],
                        self.model.Idx_config:batch['Idx_Config'],
                        self.model.X_Sites:batch['X_Sites'],
                        self.model.X_NSs:batch['X_NSs'],
                        self.model.Dropout_rate:1.0,
                        self.model.is_training:False})
                    self.hook.End_TestBatch(batch['Y']*Y_std+Y_mean,Ypt*Y_std+Y_mean)
                self.hook.End_Test()
                self.hook.End_Train(sess,saver,self.modelpath,global_step,lr)
                if self.hook.converged:
                    break
                
            