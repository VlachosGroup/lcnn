# -*- coding: utf-8 -*-
# Copyright (c) Vlachos Group, Jung Group
# GNU v3.0 license

__author__ = 'Geun Ho Gu'
__copyright__ = "Copyright 2019, Vlachos Group, Jung Group"
__version__ = "1.0"
__maintainer__ = "Geun Ho Gu"
__email__ = "googhgoo@hotmail.com"
__date__ = "July 31, 2019"

import argparse
import logging
import os 
import json

import tensorflow as tf
import numpy as np 

from lcnn.model import ModelBuilder
from lcnn.trainer import Hook, Trainer
from lcnn.data import UniversalLoader

def get_parser():
    main_parser = argparse.ArgumentParser()

    # training
    training_parser = argparse.ArgumentParser(add_help=False)
    training_parser.add_argument('modelpath',
        help='Destination of model')
    training_parser.add_argument('datapath',
        help='Data location')
    training_parser.add_argument('--batch_size', type=int,
        help='Mini-batch size for training validation,\
        prediction (default:%(default)s)',
        default=1000)
    training_parser.add_argument('--max_epoch', type=int,
        help='Maximum number of epoch (default:%(default)s)',
        default=100000)
    training_parser.add_argument('--restart', action='store_true',
        help='restart from the previous best model')
    training_parser.add_argument('--reducefrombest', action='store_true',
        help='when learning rate is reduced, restore best model')
    training_parser.add_argument('--L2Penalty', type=float,
        help='Penalty of L2norm regularization\
        (default:%(default)s)',
        default=6.74e-05)
    training_parser.add_argument('--dropout_rate', type=float,
        help='dropout_rate \
        (default:%(default)s)',
        default=1.0)
    training_parser.add_argument('--init_lr', type=float,
        help='initial learning rate of Adam optimizer \
        (default:%(default)s)',
        default=1e-1)
    training_parser.add_argument('--final_lr', type=float,
        help='final learning rate which training will terminate \
        (default:%(default)s)',
        default=1e-2)
    training_parser.add_argument('--decay_rate', type=float,
        help='exponential decay rate \
        (default:%(default)s)',
        default=0.5)
    training_parser.add_argument('--patience', type=float,
        help='Epochs without improvement in validation before termination \
        (default:%(default)s)',
        default=5000)
    training_parser.add_argument('--split', type=int, nargs=3,
        help='Split data into [train] [validation] [test] and remaining for testing',
        default=None)
    training_parser.add_argument('--seed', type=int, 
        help='seed for spliting data',
        default=None)
    
    # model
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--nconv', type=int,
        help='number of convolution \
        (default:%(default)s)',
        default=2)
    model_parser.add_argument('--feature', type=int,
        help='number of feature \
        (default:%(default)s)',
        default=44)
    model_parser.add_argument('--final_feature', type=int,
        help='number of feature \
        (default:%(default)s)',
        default=4)
    model_parser.add_argument('--cutoff', type=float,
        help='cutoff radius in Angsrtom \
        (default:%(default)s)',
        default=6.00)
    
    # evaluation
    eval_parser = argparse.ArgumentParser(add_help=False)
    eval_parser.add_argument('modelpath',
        help='Destination of model')
    eval_parser.add_argument('datapath',
        help='Data location')

    ## setup subparser structure
    cmd_subparsers = main_parser.add_subparsers(dest='mode', help='Command arguments')
    cmd_subparsers.required = True

    train_mode = cmd_subparsers.add_parser('train', help='Training help', parents=[training_parser, model_parser])
    eval_mode = cmd_subparsers.add_parser('eval', help='Eval help', parents=[eval_parser])
    
    return main_parser

def evaluate(sess,model,dataloader,resultpath):
    """ evaluate the model. Predictions, site formation energy,
    latent space are saved as json. 
    
    parameters
    ----------
    model : lcnn model.
    dataloader : DataLoader class. Testset is used.
    resultpath : path where the result will be printed.
    """
    X_mean_T,X_std_T,Y_mean_T,Y_std_T = model.GetStatistics()
    X_mean = sess.run(X_mean_T)
    X_std = sess.run(X_std_T)
    Y_mean = sess.run(Y_mean_T)
    Y_std = sess.run(Y_std_T)
    test_XY = dataloader.GetTestSet()
    Ypts = []
    latentspace = [[] for _ in model.SiteLayer]
    y_site_conts = []
    for batch in test_XY:
        (Ypt,y_site_cont,latent) = sess.run((model.Y_conf_wise,model.Y_atom_wise,model.SiteLayer),
            feed_dict={
            model.N_Sites:batch['N_Sites'],
            model.N_Sites_per_config:batch['N_Sites_per_config'],
            model.Idx_config:batch['Idx_Config'],
            model.X_Sites:batch['X_Sites'],
            model.X_NSs:batch['X_NSs'],
            model.Dropout_rate:1.0,
            model.is_training:False})
        Ypts.append(Ypt*Y_std+Y_mean)
        y_site_cont = y_site_cont*Y_std+Y_mean
        configidx = np.unique(batch['Idx_Config'])
        for i in configidx:
            y_site_conts.append(np.reshape(np.take(y_site_cont,np.where(batch['Idx_Config']==i)[0]),(-1)).tolist())
        for i,l in enumerate(latent):
            for j in configidx:
                latentspace[i].append(np.take(l,np.where(batch['Idx_Config']==j)[0],axis=0).tolist())
    Ypts = np.concatenate(Ypts).tolist()
    
    json.dump(Ypts,open(os.path.join(resultpath,'prediction.json'),'w'))
    json.dump(y_site_conts,open(os.path.join(resultpath,'site_wise.json'),'w'))
    json.dump(latentspace,open(os.path.join(resultpath,'latent.json'),'w'))

if __name__ == '__main__':
    """Main Loop 
    """
    parser = get_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    if args.mode == 'train':
        logging.info('Loading and preprocessing data... Patience is the virtue...')
        dataloader = UniversalLoader(args.modelpath,args.datapath,args.cutoff,args.split,args.batch_size,args.seed)
        logging.info('Finished.')
        json.dump(vars(args),open(os.path.join(args.modelpath,'params.json'),'w'),indent=4)
        logging.info('Training...')

        lcnn = ModelBuilder(dataloader.GetNOccupancy(),dataloader.GetNNeighbors(),
            dataloader.GetNPermutation(),dataloader.GetXMean(),dataloader.GetXStd(),
            dataloader.GetYMean(),dataloader.GetYStd(),
            args.nconv,args.feature,args.final_feature)   

        hook = Hook(os.path.join(args.modelpath,'train_log.csv'),args.restart,False,True)
        trainer = Trainer(args.modelpath,lcnn,dataloader,hook,args.L2Penalty,
                      args.dropout_rate,args.init_lr,args.final_lr,args.decay_rate,args.patience,args.max_epoch,args.restart)
        trainer.Train()
    if args.mode == 'eval':
        logging.info('Loading model...')
        params = json.load(open(os.path.join(args.modelpath,'params.json'),'r'))

        dataloader = UniversalLoader(args.modelpath,args.datapath,params['cutoff'],[0,0],params['batch_size'],params['seed'])
        
        
        lcnn = ModelBuilder(dataloader.GetNOccupancy(),dataloader.GetNNeighbors(),
            dataloader.GetNPermutation(),dataloader.GetXMean(),dataloader.GetXStd(),
            dataloader.GetYMean(),dataloader.GetYStd(),
            params['nconv'],params['feature'],params['final_feature'])
        logging.info('Evaluating...')
        sess = tf.Session()
        saver = tf.train.Saver()
        save_path = saver.restore(sess, os.path.join(args.modelpath,"bestmodel.ckpt"))
        evaluate(sess,lcnn,dataloader,args.datapath)
            
    logging.info('Finished')
