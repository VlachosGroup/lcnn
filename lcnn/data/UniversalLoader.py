# -*- coding: utf-8 -*-
# Copyright (c) Vlachos Group, Jung Group
# GNU v3.0 license

__author__ = 'Geun Ho Gu'
__copyright__ = "Copyright 2019, Vlachos Group, Jung Group"
__version__ = "1.0"
__maintainer__ = "Geun Ho Gu"
__email__ = "ghgu@kaist.ac.kr"
__date__ = "July 31, 2019"

import numpy as np
import json
import os
from collections import defaultdict
import logging

from pymatgen import Element, Structure, Molecule, Lattice
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
import networkx as nx
import networkx.algorithms.isomorphism as iso
from scipy.spatial.distance import cdist,pdist,squareform

from .Data import DataLoader
__all__ = [
        'UniversalLoader',
        'UniversalLoaderInputWriter'
        ]

# TQDM substitute for stdout
class Progress(object):
    def __init__(self, iterator , ndata, step = 1000):
        logging.getLogger().setLevel(logging.INFO)
        self.iter = iterator.__iter__()
        self.t = time()
        self.ndata = ndata
        self.step = step
        self.i = 0
        s = ' '.join(datetime.now().strftime("[%H:%M:%S]"),'0','/',self.ndata)
        logging.info(s)
        
        
    def __iter__(self):
        return self
        
    def __next__(self):
        self.i += 1
        if self.i%self.step == 0:
            s = ' '.join(datetime.now().strftime("[%H:%M:%S]"),self.i,'/',self.ndata,\
            '| %.2f/1000 sec/data |'%((time()-self.t)/self.i*self.step), '~%.2f sec left'%((self.ndata-self.i)/self.i*(time()-self.t)))
            logging.info(s)
        return next(self.iter)

def InputReader(path):
    """Read Input Files
    The input format for primitive cell is:
    
    [comment]
    [ax][ay][az][pbc]
    [bx][by][bz][pbc]
    [cx][cy][cz][pbc]
    [number of spectator site type][number of active site type]
    [os1][os2][os3]...
    [number sites]
    [site1a][site1b][site1c][site type]
    [site2a][site2b][site2c][site type]
    ...
    [number of data]
    [datum 1 name]
    ...
    
    - ax,ay, ... are cell basis vector
    - pbc is either T or F indication of the periodic boundary condition
    - os# is the name of the possible occupation state (interpretted as string)
    - site1a,site1b,site1c are the scaled coordinates of site 1
    - site type can be either S1, S2, ... or A1, A2,... indicating spectator 
        site and itx index and active site and its index respectively.
    
    Example:
    #Primitive Cell
     2.81859800e+00  0.00000000e+00  0.00000000e+00 T
    -1.40929900e+00  2.44097800e+00  0.00000000e+00 T
     0.00000000e+00  0.00000000e+00  2.55082550e+01 T
    1 1
    -1 0 1
    6
     0.00000000e+00  0.00000000e+00  9.02210000e-02 S1
     6.66666666e-01  3.33333333e-01  1.80442000e-01 S1
     3.33333333e-01  6.66666666e-01  2.69674534e-01 S1
     0.00000000e+00  0.00000000e+00  3.58978557e-01 S1
     6.66666666e-01  3.33333333e-01  4.49958662e-01 S1
     3.33333333e-01  6.66666666e-01  5.01129144e-01 A1
    653
    structure000
    structure001
    ...
    
    The input format for a data point is similar:
    
    [property value]
    [ax][ay][az]
    [bx][by][bz]
    [cx][cy][cz]
    [number sites]
    [site1a][site1b][site1c][site type][occupation state if active site]
    [site2a][site2b][site2c][site type][occupation state if active site]
    ...
    
    - property value indicates the trained value. It must start with #y=...
    
    Example:
    #y=-1.209352
     2.81859800e+00  0.00000000e+00  0.00000000e+00
    -1.40929900e+00  2.44097800e+00  0.00000000e+00
     0.00000000e+00  0.00000000e+00  2.55082550e+01
    6
     0.000000000000  0.000000000000  0.090220999986 S1
     0.500000499894  0.622008360788  0.180442000011 S1
     0.999999500106  0.666666711253  0.270892474701 S1
     0.000000000000  0.000000000000  0.361755713893 S1
     0.500000499894  0.622008360788  0.454395429618 S1
     0.000000000000  0.666667212896  0.502346789304 A1 1
     
    Parameters
    ----------
    path : input file path
    
    Returns
    -------
    list of local_env : list of local_env class
    """
    with open(path) as f:
        
        s = f.readlines()
        s = [line.rstrip('\n') for line in s]
        nl = 0
        # read comment
        if '#y=' in s[nl]:
            y = float(s[nl][3:])
            datum = True
        else:
            y = None
            datum = False
        nl += 1
        # load cell and pbc
        cell = np.zeros((3,3))
        pbc = np.array([True,True,True])
        for i in range(3):
            t = s[nl].split()
            cell[i,:] = [float(i) for i in t[0:3]]
            if not datum and t[3] == 'F':
                pbc[i] = False
            nl += 1
        # read sites if primitive
        if not datum:
            t = s[nl].split()
            ns = int(t[0])
            na = int(t[1])
            nl += 1
            aos = s[nl].split()
            nl += 1
        # read positions
        nS = int(s[nl])
        nl += 1
        coord = np.zeros((nS,3))
        st = []
        oss = []
        for i in range(nS):
            t = s[nl].split()
            coord[i,:] = [float(i) for i in t[0:3]]
            st.append(t[3])
            if datum and len(t) == 5:
                    oss.append(t[4])
            nl+=1
        # read data name
        if not datum:
            nd = int(s[nl])
            nl += 1
            datanames = []
            for i in range(nd):
                datanames.append(s[nl])
                nl += 1
    
    if datum:
        return y, cell, coord, st, oss
    else:
        return cell, pbc, coord, st, ns, na, aos, datanames
    
def UniversalLoaderInputWriter(path,y,cell,coord,st,oss):
    """Writes datum into file. 
    This can be used to print out input format of the datum you have.
   
    parameters
    ----------
    path : string. path to file for writing.
    y : float. target property value
    cell : 3 x 3. list of list of float. cell basis vectors
    coord : ns x 3. list of list of float. scaled positions of each site.
        ns is the number of sites.
    st : ns. list of string. site type for each site.
    oss : nsa. list of string. occupancy of each active site. In the order
        of appearance in coord. nsa is the number of active site. 
    """
    s = '#y=%e\n'%y
    for v in cell:
        s += '%15.8e %15.8e %15.8e\n'%(v[0],v[1],v[2])
    s+= str(len(st))+'\n'
    n =0
    for xyz,ss in zip(coord,st):
        if ss == 'S1':
            s += '%15.12f %15.12f %15.12f %s\n'%(xyz[0],xyz[1],xyz[2],ss)
        else:
            s += '%15.12f %15.12f %15.12f %s %s\n'%(xyz[0],xyz[1],xyz[2],ss,oss[n])
            n +=1
    with open(path,'w') as f:
        f.write(s)
        
class SiteEnvironment(object):
    def __init__(self,pos,sitetypes,env2config,permutations,cutoff,\
                 Grtol=0.0,Gatol=0.01,rtol = 0.01,atol=0.0, tol=0.01,grtol=0.01):
        """ Initialize site environment
        
        This class contains local site enrivonment information. This is used
        to find neighborlist in the datum (see GetMapping).
        
        Parameters
        ----------
        pos : n x 3 list or numpy array of (non-scaled) positions. n is the 
            number of atom.
        sitetypes : n list of string. String must be S or A followed by a 
            number. S indicates a spectator sites and A indicates a active 
            sites.
        permutations : p x n list of list of integer. p is the permutation 
            index and n is the number of sites.
        cutoff : float. cutoff used for pooling neighbors. for aesthetics only
        Grtol : relative tolerance in distance for forming an edge in graph
        Gatol : absolute tolerance in distance for forming an edge in graph
        rtol : relative tolerance in rmsd in distance for graph matching
        atol : absolute tolerance in rmsd in distance for graph matching
        tol : maximum tolerance of position RMSD to decide whether two 
            environment are the same
        grtol : tolerance for deciding symmetric nodes
        """
        self.pos = pos
        self.sitetypes = sitetypes
        self.activesiteidx = [i for i,s in enumerate(self.sitetypes) if 'A' in s]
        self.formula = defaultdict(int)
        for s in sitetypes:
            self.formula[s] += 1 
        self.permutations = permutations
        self.env2config = env2config
        self.cutoff = cutoff
        # Set up site environment matcher
        self.tol = tol
        # Graphical option
        self.Grtol = Grtol
        self.Gatol = Gatol
        #tolerance for grouping nodes
        self.grtol =1e-3
        # determine minimum distance between sitetypes.
        # This is used to determine the existence of an edge
        dists = squareform(pdist(pos))
        mindists = defaultdict(list)
        for i,row in enumerate(dists):
            row_dists = defaultdict(list)
            for j in range(0,len(sitetypes)):
                if i == j:
                    continue
                # Sort by bond
                row_dists[frozenset((sitetypes[i],sitetypes[j]))].append(dists[i,j])
            for pair in row_dists:
                mindists[pair].append(np.min(row_dists[pair]))
        # You want to maximize this in order to make sure every node gets an edge
        self.mindists = {}
        for pair in mindists:
                self.mindists[pair] = np.max(mindists[pair])
        # construct graph
        self.G = self._ConstructGraph(pos,sitetypes)
        # matcher options
        self._nm = iso.categorical_node_match('n','')
        self._em = iso.numerical_edge_match('d',0,rtol,0)
    
    def _ConstructGraph(self,pos,sitetypes):
        """Returns local environment graph using networkx and
        tolerance specified.
        
        parameters
        ----------
        pos: ns x 3. coordinates of positions. ns is the number of sites.
        sitetypes: ns. sitetype for each site
        
        return
        ------
        networkx graph used for matching site positions in
        datum. 
        """
        # construct graph
        G = nx.Graph()
        dists = cdist([[0,0,0]],pos - np.mean(pos,0))[0]
        sdists = np.sort(dists)
        #https://stackoverflow.com/questions/37847053/uniquify-an-array-list-with-a-tolerance-in-python-uniquetol-equivalent
        uniquedists = sdists[~(np.triu(np.abs(sdists[:,None]-sdists)<=self.grtol,1)).any(0)]
        orderfromcenter = np.digitize(dists,uniquedists)
        # Add nodes
        for i,o in enumerate(orderfromcenter):
            G.add_node(i,n=str(o)+sitetypes[i])
        # Add edge. distance is edge attribute
        dists = pdist(pos); n=0
        for i in range(len(sitetypes)):
            for j in range(i+1,len(sitetypes)):
                if dists[n] < self.mindists[frozenset((sitetypes[i],sitetypes[j]))] or\
                    (abs(self.mindists[frozenset((sitetypes[i],sitetypes[j]))] - dists[n]) <= self.Gatol + self.Grtol * abs(dists[n])):
                    G.add_edge(i,j,d=dists[n])
                n+=1
        return G
    def __repr__(self):
        s = '<' + self.sitetypes[0]+\
            '|%i active neighbors'%(len([s for s in self.sitetypes if 'A' in s])-1)+\
            '|%i spectator neighbors'%len([s for s in self.sitetypes if 'S' in s])+\
            '|%4.2f Ang Cutoff'%self.cutoff + '| %i permutations>'%len(self.permutations)
        return s
    
    def __eq__(self,o):
        """Local environment comparison is done by comparing represented site
        """
        if not isinstance(o,SiteEnvironment):
            raise ValueError
        return self.sitetypes[0] == o.sitetypes[0]
    def __ne__(self,o):
        """Local environment comparison is done by comparing represented site
        """
        if isinstance(o,SiteEnvironment):
            raise ValueError
        return not self.__eq__(o)
    
    def GetMapping(self,env,path=None):
        """Returns mapping of sites from input to this object
        
        Pymatgen molecule_matcher does not work unfortunately as it needs to be
        a reasonably physical molecule.
        Here, the graph is constructed by connecting the nearest neighbor, and 
        isomorphism is performed to find matches, then kabsch algorithm is
        performed to make sure it is a match. NetworkX is used for portability.
        
        Parameters
        ----------
        env : dictionary that contains information of local environment of a 
            site in datum. See _GetSiteEnvironments defintion in the class
            SiteEnvironments for what this variable should be.
        
        Returns
        -------
        dict : atom mapping. None if there is no mapping
        """
        # construct graph
        G = self._ConstructGraph(env['pos'],env['sitetypes'])
        if len(self.G.nodes) != len(G.nodes):
            s = 'Number of nodes is not equal.\n'
            raise ValueError(s)
        elif len(self.G.edges) != len(G.edges):
            print(len(self.G.edges),len(G.edges))
            s = 'Number of edges is not equal.\n'
            s += "- Is the data point's cell a redefined lattice of primitive cell?\n"
            s += '- If relaxed structure is used, you may want to check structure or increase Gatol\n'
            if path:
                s += path
            raise ValueError(s)
        GM = iso.GraphMatcher(self.G,G,self._nm,self._em)
        ######################## Most Time Consuming Part #####################
        ams = list(GM.isomorphisms_iter())
        # Perhaps parallelize it?
        ######################## Most Time Consuming Part #####################
        if not ams:
            s = 'No isomorphism found.\n'
            s += "- Is the data point's cell a redefined lattice of primitive cell?\n"
            s += '- If relaxed structure is used, you may want to check structure or increase rtol\n'
            if path:
                s += path
            raise ValueError(s)
        
        rmsd = []
        for am in ams: #Loop over isomorphism
            # reconstruct graph after alinging point order
            xyz = np.zeros((len(self.pos),3))
            for i in am:
                xyz[i,:] = env['pos'][am[i],:]
            R = self._kabsch(self.pos,xyz)
            #RMSD
            rmsd.append(np.sqrt(np.mean(np.linalg.norm(np.dot(self.pos,R)-xyz,axis=1)**2)))
        mini = np.argmin(rmsd)
        minrmsd = rmsd[mini]
        if minrmsd < self.tol:
            return ams[mini]
        else:
            s = 'No isomorphism found.\n'
            s += '-Consider increasing neighbor finding tolerance'
            raise ValueError(s)
        
    def _kabsch(self, P, Q):
        """Returns rotation matrix to align coordinates using
        Kabsch algorithm. 
        """
        C = np.dot(np.transpose(P), Q)
        V, S, W = np.linalg.svd(C)
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]
        R = np.dot(V, W)
        return R


class SiteEnvironments(object):
    def __init__(self,site_envs,ns,na,aos,eigen_tol,pbc,cutoff, dnames= None):
        """Initialize
        
        Use Load to intialize this class.
        
        Parameters
        ----------
        site_envs : list of SiteEnvironment object
        ns : int. number of spectator sites types
        na : int. number of active sites types
        aos : list of string. avilable occupational states for active sites
            string should be the name of the occupancy. (consistent with the input data)
        eigen_tol : tolerance for eigenanalysis of point group analysis in
            pymatgen.
        pbc : periodic boundary condition.
        cutoff : float. Cutoff radius in angstrom for pooling sites to 
            construct local environment 
        """
        self.site_envs = site_envs
        self.unique_site_types = [env.sitetypes[0] for env in self.site_envs]
        self.ns = ns
        self.na = na
        self.aos = aos
        self.eigen_tol = eigen_tol
        self.pbc = pbc
        self.cutoff = cutoff
        self.dnames = dnames
        
    def __repr__(self):
        s = '<%i active sites'%(self.na)+'|%i spectator sites'%(self.ns) +'>'
        return s
    
    def __getitem__(self, el):
        """Returns a site environment
        """
        return self.site_envs[el]
    
    def ReadDatum(self,path,cutoff_factor = 1.1):
        """Load structure data and return neighbor information
        
        Parameters
        ----------
        path : path of the structure
        cutoff_factor : float. this is extra buffer factor multiplied 
            to cutoff to ensure pooling all relevant sites. 
                
        Return
        ------
        Y : property value
        XSites : one hot encoding of the site. See DataLoader in Data.py
            for detailed instruction.
        neighborlist : s x n x p x i. s is the type of site index, 
            n is the site index, p is the permutation,
            index and i is the neighbor sites index (0 being the site itself).
            See DataLoader in Data.py for detailed instruction.
        """
        Y, cell, coord, st, oss = InputReader(path)
        # Construct one hot encoding
        XSites = np.zeros((len(oss),len(self.aos)))
        for i,o in enumerate(oss):
            XSites[i,self.aos.index(o)] = 1
        # get mapping between all site index to active site index
        alltoactive = {}
        n = 0
        for i,s in enumerate(st):
            if 'A' in s:
                alltoactive[i] = n
                n+=1
        # Get Neighbors
        ## Read Data
        site_envs = self._GetSiteEnvironments(coord,cell,st,self.cutoff*cutoff_factor,
            self.pbc,get_permutations=False,eigen_tol=self.eigen_tol)
        XNSs = [[] for _ in range(len(self.site_envs))]
        for env in site_envs:
            i = self.unique_site_types.index(env['sitetypes'][0])
            env = self._truncate(self.site_envs[i],env)
            
            # get map between two environment 
            mapping = self.site_envs[i].GetMapping(env,path)
            # align input to the primitive cell (reference)
            aligned_idx = [env['env2config'][mapping[i]] for i in range(len(env['env2config']))]
            # apply permutations
            nni_perm = np.take(aligned_idx,self.site_envs[i].permutations)
            # remove spectators
            nni_perm = nni_perm[:,self.site_envs[i].activesiteidx]
            # map it to active sites
            nni_perm = np.vectorize(alltoactive.__getitem__)(nni_perm)
            XNSs[i].append(nni_perm.tolist())
        return Y, XSites.tolist(), XNSs
    @classmethod
    def _truncate(cls,env_ref,env):
        """When cutoff_factor is used, it will pool more site than cutoff factor specifies.
        This will rule out nonrelevant sites by distance.
        """
        # Extract the right number of sites by distance
        dists = defaultdict(list)
        for i,s in enumerate(env['sitetypes']):
            dists[s].append([i,env['dist'][i]])
        for s in dists:
            dists[s] = sorted(dists[s], key= lambda x:x[1])
        siteidx = []
        for s in dists:
            siteidx += [i[0] for i in dists[s][:env_ref.formula[s]]]
        siteidx = sorted(siteidx)
        env['pos']=[env['pos'][i] for i in range(len(env['pos'])) if i in siteidx]
        
        env['pos']=np.subtract(env['pos'],np.mean(env['pos'],0))
        env['sitetypes'] = [env['sitetypes'][i] for i in range(len(env['sitetypes'])) if i in siteidx]
        env['env2config'] = [env['env2config'][i] for i in siteidx]
        del env['dist']
        return env
    
    @classmethod
    def Load(cls,path,cutoff,eigen_tol=1e-5):
        """Load Primitive cell and return SiteEnvironments
        
        Parameters
        ----------
        path : input file path
        cutoff : float. cutoff distance in angstrom for collecting local
            environment.
        eigen_tol : tolerance for eigenanalysis of point group analysis in
            pymatgen.
        
        """
        cell, pbc, coord, st, ns, na, aos, dnames = InputReader(path)
        site_envs = cls._GetSiteEnvironments(coord,cell,st,cutoff,pbc,True,eigen_tol=eigen_tol)
        site_envs = [SiteEnvironment(e['pos'],e['sitetypes'],e['env2config'],
                     e['permutations'],cutoff) for e in site_envs]
       
        ust = [env.sitetypes[0] for env in site_envs]
        usi = np.unique(ust,return_index=True)[1]
        site_envs = [site_envs[i] for i in usi]
        return cls(site_envs,ns,na,aos,eigen_tol,pbc,cutoff, dnames)
    
    @classmethod
    def _GetSiteEnvironments(cls,coord,cell,SiteTypes,cutoff,pbc,get_permutations=True,eigen_tol=1e-5):
        """Extract local environments from primitive cell
        
        Parameters
        ----------
        coord : n x 3 list or numpy array of scaled positions. n is the number 
            of atom.
        cell : 3 x 3 list or numpy array
        SiteTypes : n list of string. String must be S or A followed by a 
            number. S indicates a spectator sites and A indicates a active 
            sites.
        cutoff : float. cutoff distance in angstrom for collecting local
            environment.
        pbc : list of boolean. Periodic boundary condition
        get_permutations : boolean. Whether to find permutatated neighbor list or not.
        eigen_tol : tolerance for eigenanalysis of point group analysis in
            pymatgen.
        
        Returns
        ------
        list of local_env : list of local_env class
        """
        #%% Check error
        assert isinstance(coord,(list,np.ndarray))
        assert isinstance(cell,(list,np.ndarray))
        assert len(coord) == len(SiteTypes)
        #%% Initialize
        # TODO: Technically, user doesn't even have to supply site index, because 
        #       pymatgen can be used to automatically categorize sites.. 
        coord = np.mod(coord,1)
        pbc = np.array(pbc)
        #%% Map sites to other elements.. 
        # TODO: Available pymatgne functions are very limited when DummySpecie is 
        #       involved. This may be perhaps fixed in the future. Until then, we 
        #       simply bypass this by mapping site to an element
        # Find available atomic number to map site to it
        availableAN = [i+1 for i in reversed(range(0,118))]
        
        # Organize Symbols and record mapping
        symbols = []
        site_idxs = []
        SiteSymMap = {} # mapping
        SymSiteMap = {}
        for i,SiteType in enumerate(SiteTypes):
            if SiteType not in SiteSymMap:
                symbol = Element.from_Z(availableAN.pop())
                SiteSymMap[SiteType] = symbol
                SymSiteMap[symbol] = SiteType

            else:
                symbol = SiteSymMap[SiteType]
            symbols.append(symbol)
            if 'A' in SiteType:
                site_idxs.append(i)
        #%% Get local environments of each site
        # Find neighbors and permutations using pymatgen
        lattice = Lattice(cell)
        structure = Structure(lattice, symbols,coord)
        neighbors = structure.get_all_neighbors(cutoff,include_index=True)
        site_envs = []
        for site_idx in site_idxs:
            local_env_sym = [symbols[site_idx]]
            local_env_xyz = [structure[site_idx].coords]
            local_env_dist = [0.0]
            local_env_sitemap = [site_idx]
            for n in neighbors[site_idx]:
                # if PBC condition is fulfilled.. 
                c = np.around(n[0].frac_coords,10)
                withinPBC = np.logical_and(0<=c,c<1)
                if np.all(withinPBC[~pbc]):
                    local_env_xyz.append(n[0].coords)
                    local_env_sym.append(n[0].specie)
                    local_env_dist.append(n[1])
                    local_env_sitemap.append(n[2])
            local_env_xyz = np.subtract(local_env_xyz,np.mean(local_env_xyz,0))
            
            perm = []
            if get_permutations:
                finder = PointGroupAnalyzer(Molecule(local_env_sym,local_env_xyz),eigen_tolerance=eigen_tol)
                pg = finder.get_pointgroup()
                for i,op in enumerate(pg):
                    newpos = op.operate_multi(local_env_xyz)
                    perm.append(np.argmin(cdist(local_env_xyz,newpos),axis=1).tolist())
                
            site_env = {'pos':local_env_xyz,'sitetypes':[SymSiteMap[s] for s in local_env_sym],
                        'env2config':local_env_sitemap,'permutations':perm,
                        'dist':local_env_dist}
            site_envs.append(site_env)
        return site_envs
    
def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class UniversalLoader(DataLoader):
    def __init__(self,modelpath,datapath,cutoff,split,batch_size,seed=None,eval=False):
        """Load data
        
        See DataLoader in Data.py for detailed instruction.
        
        Parameters
        ----------
        modelpath : path where the model will be.
        datapath : path to where input.in is in.
        cutoff : float. cutoff of radius for getting local environment.Only 
            used down to 2 digits.
        split : list of two integers. Size of Train set and Validation set.
            Test set is the remaining number of set.
        batch_size : size of the batch.
        seed : random seed for spliting data points
        """
        
        
        cutoff = np.around(cutoff,2)
        if not os.path.exists(os.path.join(datapath,'data_%4.2f.json'%(cutoff))):
            input_path = 'input.in'
            # Load primitive cell
            SEnvs = SiteEnvironments.Load(os.path.join(datapath,input_path),cutoff)
            # Load datapoints
            Y = []
            XSites = []
            XNSs = []
            # parallization
            try: 
                import multiprocessing
                p = multiprocessing.Pool() 
            except:
                multiprocessing = None
                
            inputs = [os.path.join(datapath,dpath) for dpath in SEnvs.dnames]
            import random
            if seed is not None: random.seed(seed)
            random.shuffle(inputs)
            
            if multiprocessing is not None:
                gen = p.imap_unordered(SEnvs.ReadDatum,inputs)
            else:
                gen = map(SEnvs.ReadDatum,inputs)
                
            for y, xSites, xNSs in Progress(gen,ndata = len(raw_data)):
                Y.append(y)
                XSites.append(xSites)
                XNSs.append(xNSs)
                
            if multiprocessing is not None:
                p.close()
                p.join()    
                
            json.dump((Y,XSites,XNSs),open(os.path.join(datapath,'data_%4.2f.json'%(cutoff)),'w'))
        else:
            Y, XSites, XNSs = json.load(open(os.path.join(datapath,'data_%4.2f.json'%(cutoff)),'r'))
            
        # randomize data set
        if seed:
            randidx = np.random.RandomState(seed=seed).permutation(len(XSites)).tolist()
        elif eval:
            randidx = list(range(len(XSites)))
        else:
            randidx = np.random.permutation(len(XSites)).tolist()
        
        # split
        if split:
            Split = {'Train':[],'Validation':[],'Test':[]}
            for i in range(len(XSites)):
                if i < split[0]:
                    Split['Train'].append(randidx[i])
                elif i > split[0] and i < split[0] + split[1] + 1:
                    Split['Validation'].append(randidx[i])
                elif i > split[0] + split[1] and i < split[0] + split[1] + split[2] + 1:
                    Split['Test'].append(randidx[i])
            if not os.path.exists(modelpath):
                os.mkdir(modelpath)
            json.dump(Split,open(os.path.join(modelpath,'split.json'),'w'))
        elif os.path.exists(os.path.join(modelpath,'split.json')): 
            Split = json.load(open(os.path.join(modelpath,'split.json'),'r'))
        else:
            raise ValueError('--split argument or split data needs to be provided')
        
        # Compute statistics of training set
        if Split['Train']:
            XSites_train = []
            Y_train = []
            for i in Split['Train']:
                XSites_train.append(XSites[i])
                Y_train.append(Y[i])
                
            XSites_train = np.concatenate(XSites_train)
            Y_train = np.array(Y_train)
            
            X_mean = XSites_train.mean(axis=0)
            X_std = XSites_train.std(axis=0)
            Y_mean = Y_train.mean(axis=0)
            Y_std = Y_train.std(axis=0)
        else: # In the case of evaluation, it's stored with the model
            X_mean = None
            X_std = None
            Y_mean = None
            Y_std = None

            
        # separate data
        Data={}
        for setname in Split:
            batches = []
            for idxs in _chunks(Split[setname],batch_size):
                batch = {}
                batch['Y'] = [Y[i] for i in idxs]
                XSites_batch = [XSites[i] for i in idxs]
                XNSs_batch = [XNSs[i] for i in idxs]
                batch['N_Sites'], batch['N_Sites_per_config'], \
                batch['Idx_Config'], batch['X_Sites'], batch['X_NSs'] = \
                    _FlattenInput(XSites_batch,XNSs_batch)
                batches.append(batch)
            Data[setname] = batches
        # neighborlist : s x n x p x i.  is the type of site index, 
        #n is the site index, p is the permutation,
        #index and i is the neighbor sites index (0 being the site itself)
        super().__init__(Data,X_mean,X_std,Y_mean,Y_std,len(XSites[0][0]),\
              [len(x[0]) for x in XNSs[0]],\
              [len(x[0][0]) for x in XNSs[0]])
        

  
def _FlattenInput(X_Sites,X_NSs):
    """This method transform the input data that is easy to interpret for 
    human to a format that is easy to process for tensorflow. Returned 
    values are used as some of the input in feed_dict of run function.
    
    Parameters
    ----------
    Y : property value
    XSites : d x n x t. One hot encoding representation of a site.
        d is the datum index, n is the site index, t is the site type index
    X_NSs : d x s x n x p x i. neighbor site index.
        s is the type of site index (central site), 
        p is the permutation index, and i is the neighbor sites index 
        (0 being the site itself)
      
    Returns
    -------
    N_Sites: list of integer
      each integer indicates number of sites in each configuration. Used to compute 
      per site formation energy
      
    Idx_config: list of integers (in numpy array)
      In the algorithm, Site layer is flattened over each data points. This
      way, we avoid using padding, and having an upper limit for the
      maximum number of sites. calculations are faster, too. To do this, we
      need data index for each site. This vector contains that information.
    
    X_Sites: numpy 2D matrix
      Site layer flattened over data index
    
    X_NSs: s x n x p x i numpy 4D matrix. Flattened over data
    
    Note
    --------
    These returned values can be inputted into the model in feed_dict as
    feed_dict={N_Sites:N_Sites,Idx_config:Idx_config,X_Sites:X_Sites,X_NSs:X_NSs}
    """
    # initialize
    ## Convert it to numpy for easy indexing
    X_Sites_PreInt = X_Sites
    X_Sites = np.array([np.array(x_sites) for x_sites in X_Sites_PreInt])
    
    X_NSs_PreInt = X_NSs 
    X_NSs = []
    for datum in X_NSs_PreInt:
        new_datum = []
        for site_type in datum:
            new_datum.append(np.array(site_type))
        X_NSs.append(new_datum)
        
    # number of sites for each datum 
    N_Sites_per_config = [] 
    for datum in X_Sites:
        N_Sites_per_config.append(len(datum))
    Idx_config = []
    for i in range(len(X_Sites)):
        Idx_config.append(np.repeat(i,len(X_Sites[i])))
    Idx_config = np.concatenate(Idx_config).reshape(-1)
    Idx_config = np.expand_dims(Idx_config,axis=1)

    # Flattened Sites
    X_Sites = np.concatenate(X_Sites)

    # Change nearest neighbor indexing for flattening
    nsite_sum = 0
    for i,nsite in enumerate(N_Sites_per_config):
        datum = X_NSs[i]
        new_datum = [sitetype + nsite_sum for sitetype in datum]
        nsite_sum  += nsite
        X_NSs[i] = new_datum
    
    # Flatten nearest neighbor
    X_NSs_flattened = [[] for _ in range(len(X_NSs[0]))]
    for datum in X_NSs:
        for j,sitetype in enumerate(datum):
            X_NSs_flattened[j].append(sitetype)
    X_NSs_flattened = [np.concatenate(sitetype) for sitetype in X_NSs_flattened]
        
    N_Sites = X_Sites.shape[0]
    return N_Sites, N_Sites_per_config, Idx_config, X_Sites, X_NSs_flattened
