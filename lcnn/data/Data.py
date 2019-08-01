# -*- coding: utf-8 -*-
# Copyright (c) Vlachos Group, Jung Group
# GNU v3.0 license

__author__ = 'Geun Ho Gu'
__copyright__ = "Copyright 2019, Vlachos Group, Jung Group"
__version__ = "1.0"
__maintainer__ = "Geun Ho Gu"
__email__ = "ghgu@kaist.ac.kr"
__date__ = "July 31, 2019"

__all__ = [
        'DataLoader'
        ]

class DataLoader(object):
    def __init__(self,Data,X_mean,X_std,Y_mean,Y_std,NOccupancy,NPermutation,NNeighbors):
        """Data Loader for covenient data handling. 
        
        If you are building your own data loader, inherit from this class.
        UniversalLoader should be applicable to any system of your interest,
        but it does take time to preprocess data as it uses position 
        matching and graph theory. Also, the UniversalLoader works well for 
        non-relaxed positions, but not for relaxed structure. These could 
        be improved by making your own loader.
        
        During training, trainer code use this to load data.
        
        Parameters
        ----------
        Data : Dictionary that contains batches for 'Validation', 'Train', and 'Test'.
            Each key returns list of batches, where each batch is a dictionary with
            -batch['Y'] : nd. list of target properties.  nd is the number of data in the batch
            -batch['N_Sites'] : integer. total number of sites in the batch. (not config)
            -batch['N_Sites_per_config'] : nd. list of integers. number of sites for each config.
            -batch['Idx_Config'] : ns. list of integers for each site. ns is the number of site
                In the algorithm, Site layer is flattened over each data points. This
                way, we avoid using padding, and having an upper limit for the
                maximum number of sites. calculations are faster, too. To do this, we
                need data index for each site. This vector contains that information.
            -batch['X_Sites'] : ns x no. one hot encodding of the site layer. no is the number of 
                possible occupancy states.
            -batch['X_NSs'] : nst x nsi x npi x j. neighbor list. (Index in the batch['X_Sites']).
                nst is the number of site type. nsi is the number of site in the site type i.
                npi is the number of permutation in site type i. j is the neighbor index, 
                where 0 is the index of site itself.
        X_mean : no. list of mean of the site layer by column. used for normalization.
        X_std : no. list of standard deviation of the site layer by column.
        Y_mean : 1. mean of the target property
        Y_std : 1. standard deviation of the target property
        """
        # Must be train set statistics
        self.X_mean = X_mean
        self.X_std = X_std
        self.Y_mean = Y_mean
        self.Y_std = Y_std
        self.Data = Data
        self.NOccupancy = NOccupancy
        self.NPermutation = NPermutation
        self.NNeighbors = NNeighbors
        
    def GetNOccupancy(self):
        """Return number of possible occupancy state.
        """
        return self.NOccupancy

    def GetNPermutation(self):
        """Return list of number of permutations for each site.
        """
        return self.NPermutation

    def GetNNeighbors(self):
        """Return list of number of neighbors for each site (including the site itself).
        """
        return self.NNeighbors

    def GetXMean(self):
        """Return the mean values for site layer
        """
        return self.X_mean
    
    def GetXStd(self):
        """Return the std values for site layer
        """
        return self.X_std
    
    def GetYMean(self):
        """Return the mean value of targetproperties
        """
        return self.Y_mean
    
    def GetYStd(self):
        """Return the std value of targetproperties
        """
        return self.Y_std
    
    def GetTrainSet(self):
        """Return batches for the training set
        """
        return self.Data['Train']
    
    def GetValidationSet(self):
        """Return batches for the validation set
        """
        return self.Data['Validation']
    
    def GetTestSet(self):
        """Return batches for the test set
        """
        return self.Data['Test']
      