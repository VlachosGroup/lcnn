Lattice Convolutional Neural Network (LCNN)
===========================================
The lattice convolutional neural network (LCNN) is a Python library for deep learning of lattice system developed by the Vlachos group at the University of Delaware and Jung group at KAIST. The model has been built in hope to improve upon cluster expansion methods,  linear regression based on the clusters in the lattice. The LCNN performs better than the cluster expansion based methods with sufficient number of data points. See below for the documentations.

Developers
----------
Geun Ho Gu (ghgu@kaist.ac.kr)

Dependencies
------------
-  Python3
-  Numpy
-  Tensorflow (>1.10)
-  Pymatgen
-  Scipy
-  Networkx

Installation
------------
1. Clone this repository:
```
git clone https://github.com/VlachosGroup/lcnn
```
2. Run the N,NO/Pt(111) example to test:
```
cd lcnn/example/Pt_N_NO_single_set_training
bash Run.sh
```
3. If you see "Finished," you are all set!
    
Publications
------------
If you use this code, please cite:

Jonathan Lym, Geun Ho Gu, Yousung Jung, Dionisios G. Vlachos, "Lattice Convolutional Neural Network Modeling of Adsorbate Coverage Effects" *Journal of Physical Chemistry C* (accepted) DOI:10.1021/acs.jpcc.9b03370


Getting Started
===============
Please submit issues to this github page for any mistakes, or improvements.

Let's apply to my system right now!
-----------------------------------
LCNN package is a very lightweight package that has a very simple interface. Training, validation, testing, and model usage can be all done with lcnn_run.py in lcnn/bin folder. 

### Preparing LCNN input
We implemented a simple input format applicable to any lattice. A working example is in lcnn/example/Pt_N_NO_single_set_training/data folder. Our data loader requires the information on the primitive cell in a file called "input.in". "input.in" has a format of:
```
[comment]
[ax] [ay] [az] [pbc]
[bx] [by] [bz] [pbc]
[cx] [cy] [cz] [pbc]
[number of spectator site type] [number of active site type]
[os1] [os2] [os3]...
[number sites]
[site1a] [site1b] [site1c] [site type]
[site2a] [site2b] [site2c] [site type]
...
[number of data]
[path to datum 1]
...
```
-  ax,ay, ... are primitive cell basis vector.
-  pbc is either T or F, indicating the periodic boundary condition.
-  os# is the name of the possible occupation state (interpretted as string). All site types share the same set of possible occupancy state.
-  site1a, site1b, site1c are the scaled coordinates of site 1.
-  site type can be either S1, S2, ... or A1, A2,... indicating spectator site and its index, and active site and its index respectively.
Example:
```
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
648
structure000
structure001
...
```
The input format for a data point is similar:
```
[property value]
[ax] [ay] [az]
[bx] [by] [bz]
[cx] [cy] [cz]
[number sites]
[site1a] [site1b] [site1c] [site type] [occupation state if active site]
[site2a] [site2b] [site2c] [site type] [occupation state if active site]
...
```
- property value indicates the trained value. It must start with #y=...
Example:
```
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
```
**Important Note:** The code extracts site local environment from the primitive cell, and perform graph matching and distance matching to find neighbor list in data. It is **highly recommended** that data are supercells of the primitive cell, specifically, the coordinates of sites in data are perfect, non-deviated lattice. If you are using relaxed structure, you can play with graph matching tolerance in the initialization of SiteEnvironment class at lcnn/data/UniversalLoader.py, but it may not work well depending on your system.

**Additional Note:**
The current method implements graph matching, and diagonalization to find neighbors, so it can take some time to preprocess data to produce representation for LCNN. Implementing neighbor finding algorithm specific to your system can save computation time (see lcnn/data/Data.py for additional instruction for this). The current implementation can process ~1000 data points within an hour. 

### Training and Evaluating
This instruction is also provided when lcnn_run.py is executed without arguments. To train the model with the default setting, you can simply use:
```
python lcnn_run.py train [modelpath] [datapath] --split [train size] [validation size]
```
Here, train indicates that you are training a model, modelpath indicates where the result will be, datapath is the path of the folder containing "input.in", and train size and validation size indicates the size of training, and validation set, where the rest of the set will be used as the test set. The code will split the data randomly and begin training. If you would like to make your own split, produce "split.json" in the model path, and don't provide --split option. "split.json" needs to be a dictionary with keys, "Test", "Train", "Validation", elements of which should be lists of data index (see "split.json" in the example folder). There are many other hyperparameters you can set:

- --batch_size [int] : size of batch in training
- --max_epoch [int] : maximum number of epoch 
- --restart : load the best model before training
- --reducefrombest : whether to use the previous best model when reducing the learning rate
- --dropout_rate [float] : dropout rate
- --final_lr [float] : learning rate below with the training will terminate
- --decay_rate [float] : learning decay ratio on plateau
- --patience [int] : number of additional epoch to run on the plateau before reducing the learning rate
- --seed [int] : random seed
- --nconv [int] : number of convolution
- --feature [int] : number of features used during convolution
- --final_feature [int] : number of features used during site-wise activation
- --cutoff [float] : cutoff radius in angstrom for considering the local environment
- --L2Penalty [float]: L2 norm penalty value

Alternatively, you can use your trained model to evaluate by:
```
python lcnn_run.py eval [modelpath] [datapath]
```
It will load up the model in [modelpath] and evaluate the entire the data set as the test set, and print out site contribution, latent space, and predicted property in [datapath] in the order in "input.in". 

**Additional Note:**
In the paper, we described local environment size as the maximum number of edges (in terms of graph theory) from the site. Here, we implemented radius cutoff (--cutoff option). This option defines the radius of which local environment will be pulled.

### Hyperparameter Optimization
We provided a simple hyperparameter optimization routine using random sampling in example/Pt_N_NO_hyperparameter_optimization/hyperparameter_optimization.py. If you have your own system, simply swap out the input in the data folder and change ndata in the python file, and adjust hyperparameter sampling range accordingly. 
