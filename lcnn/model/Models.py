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
import numpy as np

__all__ = [
        'ModelBuilder',
        ]

class ModelBuilder(object):
    def __init__(self,n_occupancy,n_neighbor_sites_list,n_permutation_list,\
                 X_mean, X_std, Y_mean, Y_std,
                 n_conv=1,n_feature=150,sitewise_n_feature=25,
                 ):
        """
        Model builder. 
        
        Parameters
        ----------
        n_occupancy: int. number of possible occupancy
        n_neighbor_sites_list: list of int. Number of neighbors of each site. 
        X_mean : no. list of float. Mean of the one hot encoding matrix by column.
            no is the number of occupancy state.
        X_std : no. list of float. Standard deviation of the one hot encoding matrix by column.
        Y_mean : float. Mean of target property
        Y_std : float. Standard deviation of target property
        nconv: int. number of convolutions performed
        n_feature: int. number of feature for each site
        sitewise_n_feature: int. number of features for atoms for site-wise activation
        """
        tf.Graph()
        tf.reset_default_graph()
        # Not to be used within the graph.. Used in Trainer. 
        if X_mean is None: # None is used when evaluation mode is activated
            self.X_mean = tf.get_variable('X_mean',
                shape=(n_occupancy,),trainable=False)
            self.X_std = tf.get_variable('X_std',
                shape=(n_occupancy,),trainable=False)
            self.Y_mean = tf.get_variable('Y_mean',
                shape=(),trainable=False)
            self.Y_std = tf.get_variable('Y_std',
                shape=(),trainable=False)
        else:
            self.X_mean = tf.get_variable('X_mean',
                initializer=tf.convert_to_tensor(np.array(X_mean,dtype=np.float32)),
                trainable=False)
            self.X_std = tf.get_variable('X_std',
                initializer=tf.convert_to_tensor(np.array(X_std,dtype=np.float32)),
                trainable=False)
            self.Y_mean = tf.get_variable('Y_mean',
                initializer=tf.convert_to_tensor(np.array(Y_mean,dtype=np.float32)),
                trainable=False)
            self.Y_std = tf.get_variable('Y_std',
                initializer=tf.convert_to_tensor(np.array(Y_std,dtype=np.float32)),
                trainable=False)
        
        
        self.Y = tf.placeholder(dtype=tf.float32, shape = [None])
        self.Idx_config = tf.placeholder(dtype=tf.int32, shape = [None,1])
        self.N_Sites = tf.placeholder(dtype=tf.int32, shape = ())
        self.N_Sites_per_config = tf.placeholder(dtype=tf.float32, shape = [None])
        self.X_Sites = tf.placeholder(dtype=tf.float32, shape = [None,n_occupancy])
        
        
        self.X_NSs = []
        for n_neighbor_sites,n_permutation in zip(n_neighbor_sites_list,n_permutation_list):
            self.X_NSs.append(tf.placeholder(dtype=tf.int32, shape = [None,n_permutation,n_neighbor_sites]))
        self.X_NSs = tuple(self.X_NSs)
        self.Lambda = tf.placeholder(dtype=tf.float32, shape = [])
        self.Dropout_rate = tf.placeholder(dtype=tf.float32, shape = [])
        self.is_training = tf.placeholder(dtype=tf.bool, shape = [])
        
        """ initiaite Model Helper """
        self.ModelHelper = CNNHelper(dropout_rate = self.Dropout_rate, UseBN=True,is_training=self.is_training)

        """ keeps tracks of site layer """
        self.SiteLayer = [self.X_Sites]

        """ Featurize one-hot-encoding """
        #self.SiteLayer.append(self.ModelHelper.Atom_Wise_Linear(self.X_Sites,n_feature))
        
        """ Perform Convolution """
        for i in range(0,n_conv):
            self.SiteLayer.append(self.ModelHelper.LCNNConvolution(self.SiteLayer[-1],self.X_NSs,self.N_Sites,depth=n_feature))    
        self.SiteLayer.append(self.ModelHelper.Atom_Wise_Convolution(self.SiteLayer[-1],depth=sitewise_n_feature))
        
        """ Perform Linear Multiplication """            
        self.Y_atom_wise = tf.expand_dims(tf.reduce_sum(self.ModelHelper.Atom_Wise_Linear(self.SiteLayer[-1]),axis=1),axis=1)
        
        """ Sum up contributions """
        self.Y_conf_wise = self.ModelHelper.Unflatten(self.Y_atom_wise,self.Idx_config,self.N_Sites_per_config)
        
        """ Loss + L2 norm """
        Params = self.ModelHelper.Params
        Params = tf.concat(Params,0)
        self.Loss = tf.losses.mean_squared_error(labels=self.Y, predictions=self.Y_conf_wise) + self.Lambda*tf.norm(Params)
    
    def GetStatistics(self):
        """Returns statistics for site layer and target property. Used in trainer in Train.py
        """
        return self.X_mean,self.X_std,self.Y_mean,self.Y_std
    


class CNNHelper(object):
    """This code is a helper for constructing CNN model.
    """
    def __init__(self,dropout_rate = tf.constant(1,dtype=tf.float32) ,UseBN = False, is_training = None,e = 1e-3,initializer=tf.contrib.layers.xavier_initializer()):
        """
        Initialize a CNN model builder. For the low level tensorflow models, 
        keeping track of model weights can be difficult. This code automates
        this process. 
        
        parameters
        ----------
        dropout_rate: 0D tensor. drop out rate
        UseBN: boolean . Flag for using batch normalization. Recommended to turn it on.
        is_training: 0D placeholder boolean tensor. Whether training or not. Used for batch norm
        e: float. This is a offset value for batch normalization for convergence
        initializer: tensorflow intializer class. tensorflow initializer for weights. Default(xavier) works well
        """
        if UseBN and is_training == None:
            raise NotImplemented('When Batch normalization is used, is_training boolean placeholder must be provided')
        self.e = e
        self.init = initializer
        self.UseBN = UseBN
        self._nCNNLayer = 0  # Number of CNN layers
        self.dropout_rate = dropout_rate
        self.Params = []
        self.is_training = is_training
        
   
    def LCNNConvolution(self,X_Sites, X_NSs, N_Sites, depth=1):
        """Construct convolution layer
        
        parameters
        ----------
        X_Sites: 2D tensor of shape (?,one hot encoding classification size)
            Site layer flattened over data index
        
        X_NSs: 3D tensor list of (?,number of permutation, 1 (site itself) + number of neighbors)
              given that the shape is (x,y,z), x is the site index which has been
              flattened over data, y is the permutation index, and z is the
              site itself index and neighbor site index. 
        
        depth: int. depth/feature of convolution
    
        Return
        ----------
        X_Sites: 2D tensor
          Convoluted tensor
        """
        
        with tf.variable_scope("layer"+str(self._nCNNLayer)):
            indices = []
            updates = []
            for i in range(0,len(X_NSs)): # iterating over each site type
                
                with tf.variable_scope("site"+str(i)):
                    X_NSs_i = X_NSs[i]
                    # Apply permutations
                    # X_Sites_Permed : R^Number of Site x Permutation Index x Neighbor Indicex x Depth
                    X_Sites_Permed = tf.gather(X_Sites,X_NSs_i,axis=0)
                    # X_Sites_Concatenated : R^Number of Site x Permutation Index x (Neighbor Indicex x Depth)
                    X_Sites_Concatenated = tf.reshape(X_Sites_Permed,[-1,X_Sites_Permed.shape[1],X_Sites_Permed.shape[2]*X_Sites_Permed.shape[3]])
                    ## Filter shape: [filter height, in_channels, out_channels] 
                    ## Stride shape: Filter height
                    w = tf.get_variable("w",[1,X_Sites_Concatenated.shape[2],depth],dtype=tf.float32,initializer=self.init)
                    self.Params.append(tf.reshape(w,[-1]))
                    b = tf.get_variable("b",[depth],dtype=tf.float32,initializer=self.init)
                    self.Params.append(tf.reshape(b,[-1]))
        
                    Conv1d = tf.nn.conv1d(X_Sites_Concatenated,w,1,"SAME")
                    Conv1dB = tf.nn.bias_add(Conv1d,b)
                    ## Batch Normalization
                    # See https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/38320613#38320613
                    if self.UseBN:
                        u = tf.get_variable("u",[1,1,depth],dtype=tf.float32,initializer=self.init)
                        v = tf.get_variable("v",[1,1,depth],dtype=tf.float32,initializer=self.init)
                        def updateuv():
                            ut,vt = tf.nn.moments(Conv1dB,[0,1],keep_dims=True)
                            return (u.assign(ut),v.assign(vt))
                        u,v = tf.cond(self.is_training,updateuv,lambda: (u,v))
        
                        o = tf.get_variable("o",[1,1,depth],dtype=tf.float32,initializer=self.init)
                        self.Params.append(tf.reshape(o,[-1]))
                        s = tf.get_variable("s",[1,1,depth],dtype=tf.float32,initializer=self.init)
                        self.Params.append(tf.reshape(s,[-1]))
                        Conv1dBN = tf.nn.batch_normalization(Conv1dB,u,v,o,s,self.e)
                    X_Sites1 = activation(Conv1dBN)
                    X_Sites2 = tf.nn.dropout(X_Sites1,self.dropout_rate,noise_shape=[1,1,depth]) # Dropout
                    X_Sites3 = tf.reduce_sum(X_Sites2,axis=1) # Sum over each permutations
                    indices.append(X_NSs_i[:,0,0])
                    updates.append(X_Sites3)
            # gather over multiple site types
            indices = tf.expand_dims(tf.concat(indices,axis=0),1)
            updates = tf.concat(updates,axis=0)
            NewSiteLayer = tf.scatter_nd(indices,updates,[N_Sites,depth])
            self._nCNNLayer += 1
            return NewSiteLayer
    

    def Atom_Wise_Linear(self,X_Sites, depth=1):
        """
        Perform Matrix Multiplication
        
        parameters
        ----------
        X_Sites: 2D tensor of shape (?,one hot encoding classification size)
            Site layer flattened over data index
        
        depth: int. depth/feature of convolution
    
        return
        ------
        X_Sites: 2D tensor. Convoluted tensor
        """
        with tf.variable_scope("layer"+str(self._nCNNLayer)):
            ## Filter shape: [filter height, in_channels, out_channels] 
            ## Stride shape: Filter height
            w = tf.get_variable("w",[X_Sites.shape[1],depth],dtype=tf.float32,initializer=self.init)
            self.Params.append(tf.reshape(w,[-1]))
            b = tf.get_variable("b",[depth],dtype=tf.float32,initializer=self.init)
            self.Params.append(tf.reshape(b,[-1]))

            X_Sites = tf.matmul(X_Sites,w)
            X_Sites = tf.nn.bias_add(X_Sites,b)

            self._nCNNLayer += 1
            return X_Sites


    def Atom_Wise_Convolution(self,X_Sites, depth=1):
        """
        Perform self convolution to a site layer.
        
        parameters
        ----------
        X_Sites: 2D tensor of shape (?,one hot encoding classification size)
            Site layer flattened over data index
        depth: int. depth of convolution
        """
        
        with tf.variable_scope("layer"+str(self._nCNNLayer)):
            ## Filter shape: [filter height, in_channels, out_channels] 
            ## Stride shape: Filter height
            w = tf.get_variable("w",[X_Sites.shape[1],depth],dtype=tf.float32,initializer=self.init)
            self.Params.append(tf.reshape(w,[-1]))
            b = tf.get_variable("b",[depth],dtype=tf.float32,initializer=self.init)
            self.Params.append(tf.reshape(b,[-1]))

            Conv = tf.matmul(X_Sites,w)
            Conv = tf.nn.bias_add(Conv,b)
            ## Batch Normalization
            if self.UseBN:
                u = tf.get_variable("u",[1,depth],dtype=tf.float32,initializer=self.init)
                v = tf.get_variable("v",[1,depth],dtype=tf.float32,initializer=self.init)
                def updateuv():
                    ut,vt = tf.nn.moments(Conv,[0],keep_dims=True)
                    return (u.assign(ut),v.assign(vt))
                u,v = tf.cond(self.is_training,updateuv,lambda: (u,v))
                
                o = tf.get_variable("o",[1,depth],dtype=tf.float32,initializer=self.init)
                self.Params.append(tf.reshape(o,[-1]))
                s = tf.get_variable("s",[1,depth],dtype=tf.float32,initializer=self.init)
                self.Params.append(tf.reshape(s,[-1]))
                Conv = tf.nn.batch_normalization(Conv,u,v,o,s,self.e)
            X_Sites = activation(Conv)
            X_Sites = tf.nn.dropout(X_Sites,self.dropout_rate,noise_shape=[1,depth]) # Dropout
            
            self._nCNNLayer += 1
            return X_Sites

    def SoftMax(self,X_Sites,depth = 1):
        """Perform Convolution to a site layer.
        
        parameters
        ----------
        X_Sites: 2D tensor of shape (?,one hot encoding classification size)
            Site layer flattened over data index
        depth: int. depth/features of the softmax
          
        return
        ------
        X_Sites: 2D tensor. softmaxed sitelayer
        """
        with tf.variable_scope("SoftMax"):
            w = tf.get_variable("w",[X_Sites.shape[1],depth],dtype=tf.float32,initializer=self.init)
            self.Params.append(tf.reshape(w,[-1]))
            b = tf.get_variable("b",[depth],dtype=tf.float32,initializer=self.init)
            self.Params.append(tf.reshape(b,[-1]))
            X_Sites = tf.matmul(X_Sites,w)
            X_Sites = tf.nn.bias_add(X_Sites,b)
            if self.UseBN:
                u = tf.get_variable("u",[1,depth],dtype=tf.float32,initializer=self.init)
                v = tf.get_variable("v",[1,depth],dtype=tf.float32,initializer=self.init)
                def updateuv():
                    ut,vt = tf.nn.moments(X_Sites,[0],keep_dims=True)
                    return (u.assign(ut),v.assign(vt))
                u,v = tf.cond(self.is_training,updateuv,lambda: (u,v))
                
                o = tf.get_variable("o",[1,depth],dtype=tf.float32,initializer=self.init)
                self.Params.append(tf.reshape(o,[-1]))
                s = tf.get_variable("s",[1,depth],dtype=tf.float32,initializer=self.init)
                self.Params.append(tf.reshape(s,[-1]))
                X_Sites = tf.nn.batch_normalization(X_Sites,u,v,o,s,self.e)
            X_Sites = tf.nn.softmax(X_Sites,1)
            return X_Sites
    
    def Unflatten(self,X_Sites,Idx_config,N_Sites):
        """This function unflattens the flattened site layer by summation. Then
        the unflattened values are dvidided by number of sites. 
        
        parameters
        ----------
        X_Sites: 2D tensor. Site layer flattened over data index
        Idx_config: 1D tensor of integer. In the algorithm, Site layer is 
            flattened over each data points. This way, we avoid using padding, 
            and having an upper limit for the maximum number of sites. calculations 
            are faster, too. To do this, we need data index for each site.
            This vector contains that information.
          
        N_Sites: 1D tensor. Each integer indicates number of sites in each
            configuration. Used to compute per site formation energy
          
        return
        ------
        ConfigLayer: 1D tensor. Unflattened per each datum
        
        """
        # All the other failed attempts to densifying X_Sites
#        ConfigLayer = tf.Variable(tf.zeros([Idx_config[-1]+1,X_Sites.shape[1]]), tf.float32)
#        ConfigLayer = tf.Variable(tf.zeros([N_Sites.get_shape()[0],X_Sites.shape[1]]), tf.float32)
#        ConfigLayer = tf.fill([Idx_config[-1]+1,X_Sites.shape[1]],0.0)
#        ConfigLayer = tf.scatter_add(ConfigLayer,Idx_config,X_Sites)
#        ConfigLayer = tf.scatter_nd(Idx_config,X_Sites,[Idx_config[-1]+1,X_Sites.shape[1]])
#        ConfigLayer = tf.scatter_nd(Idx_config,X_Sites,[N_Sites.shape[0],X_Sites.shape[1]])
#        ConfigLayer = tf.scatter_nd(Idx_config,X_Sites,[Idx_config[-1]+1,tf.constant([1])])
#        ConfigLayer = tf.scatter_nd(Idx_config,X_Sites,[Idx_config[-1]+1,[1]])
#        ConfigLayer = tf.scatter_nd(Idx_config,X_Sites,Idx_config[-1]+1)
#        ConfigLayer = tf.scatter_nd(Idx_config,X_Sites,[1,1])
        ConfigLayer = tf.scatter_nd(Idx_config,X_Sites,[tf.reshape(Idx_config[-1]+1,[]),X_Sites.shape[1]])
        ConfigLayer = tf.reshape(ConfigLayer,[-1])
        ConfigLayer = tf.div(ConfigLayer,N_Sites)
        return ConfigLayer
        
def activation(tensor):
    """Return activation function"""
    return tf.nn.softplus(tensor)-np.log(2.0)