# Benchmarking-Surface-DL
Code for "Benchmarking Geometric Deep Learning for Cortical Segmentation and Neurodevelopmental Phenotype Prediction"


# Acknowledgements:
In this benchmarking paper, we utilise and/or reproduce code from a number of different sources and models, for which we are very grateful.
We thank the authors of the following papers and/or codebases:

[Pytorch Geometric](https://arxiv.org/abs/1903.02428)\
[Spherical CNNs (S2CNN)](https://arxiv.org/abs/1801.10130) [https://github.com/jonkhler/s2cnn](https://github.com/jonkhler/s2cnn)\
[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
(ChebNet)](https://arxiv.org/abs/1606.09375)\
[Semi-Supervised Classification with Graph Convolutional Networks (GCNConvNet)](https://arxiv.org/abs/1609.02907)\
[Spherical CNNs on Unstructured Grids (UG-SCNN)](https://arxiv.org/abs/1901.02039) [https://github.com/maxjiang93/ugscnn](https://github.com/maxjiang93/ugscnn)\
[Spherical UNet](https://arxiv.org/abs/1904.00906) [https://github.com/zhaofenqiang/Spherical_U-Net](https://github.com/zhaofenqiang/Spherical_U-Net)

# Data 

We provide processed and curated dataset from the [Developing Human Connectome Project (dHCP)](https://biomedia.github.io/dHCP-release-notes/index.html) available [here](https://gin.g-node.org/lzjwilliams/geometric-deep-learning-benchmarking), subject to the [dHCP data sharing agreement](http://www.developingconnectome.org/data-release/second-data-release/open-access-dhcp-data-terms-of-use-version-4-0_2019-05-23/).

**alternatively...dataprocess**


# Code Usage

To use the code, first create a conda environment using the environment.yml file by running:
```
conda env create -f environment.yml
```
Please note that this may cause issues with non-Windows users, or those running different versions of CUDA or python, for example. In this case, please download the required modules from their respective sources.

To run the code, all file locations must be changed as appropriate. 

# Generating Warps

100 ico 2 warps are provided in the ico_2_warps folder. To generate new icosphere 2 warps, run *generating_warps.py* found in the folder ico_2_warps.


# Regression Experiments

For all regression experiments excluding UG-SCNN, please utilise the regression experiments folder. For these experiments, experimental parameters must be set in the text files found in the *params* folder; instructions on how to do this are in the README-params file found in the folder. 
Some example experiments are provided for simplicity. 

To run a graph model experiment with params in file *experiment_1.txt* from the terminal, input:
```
python train_graph.py @params/experiment_1.txt
```
For non graph models use:
 ```
 python train.py @params/experiment_1.txt
 ```
 
Results will be automatically saved in the appropriate results folder
