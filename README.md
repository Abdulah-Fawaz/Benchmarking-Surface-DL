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

# Data & Data Processing

We provide processed and curated dataset from the [Developing Human Connectome Project (dHCP)](https://biomedia.github.io/dHCP-release-notes/index.html) available [here](https://gin.g-node.org/lzjwilliams/geometric-deep-learning-benchmarking), subject to the [dHCP data sharing agreement](http://www.developingconnectome.org/data-release/second-data-release/open-access-dhcp-data-terms-of-use-version-4-0_2019-05-23/).

Provided at the above link are the temaplte and native cortical surface features and segmentation labels. 
The cortical surface features included are those used the benchmarking; myelination, curvature, sulcal depth and corrected cortical thickness.

To generate the warps, the processing shell script is provided at *example_post_processing.sh* and can be run from the terminal.
First, the human connectome workbench software is required to run the resampling. It can be downloaded from [https://www.humanconnectome.org/software/get-connectome-workbench](https://www.humanconnectome.org/software/get-connectome-workbench)

Then the script must be modified with the correct relevant file paths.
To do this, one must first download the following and set their precise paths in the script:

1) the symmetric template must be downloaded from [https://brain-development.org/brain-atlases/atlases-from-the-dhcp-project/cortical-surface-template/](https://brain-development.org/brain-atlases/atlases-from-the-dhcp-project/cortical-surface-template/)
2) the 6th level icosahedron warps downloaded from the onedrive repository [here](https://emckclac-my.sharepoint.com/:f:/g/personal/k1812201_kcl_ac_uk/EluWzKNeKd5CmMqGc1n1cKcBwe2n2yU7CJrzoD_0u8r_7g)
3) the 6th level icosahedron which is found in the icosahedrons folder of this repository under file *ico-6.surf.gii*
4) the location of the dHCP Dataset as downloaded from above 

After correctly downloading all the required files and setting their paths in the script, running the shell will produce the complete warped dataset for training.

# Code Usage

To use the code, first create a conda environment using the environment.yml file by running:
```
conda env create -f environment.yml
```
Please note that this may cause issues with non-Windows users, or those running different versions of CUDA or python, for example. In this case, please download the required modules from their respective sources.

To run the code, all file locations must be changed as appropriate. 


# Regression Experiments

For all regression experiments **excluding UG-SCNN**, experimental parameters must be set in the text files found in the *params* folder; instructions on how to do this are in the README-params file found in the folder. **This includes both the warped and unwarped data directories**.
Some example experiments are provided for simplicity. 

To run a graph model experiment with params in file *ChebNet_NoPool_BirthAgeConfounded_Rotated_Native* from the terminal, input:
```
python train_graph.py @params/ChebNet_NoPool_BirthAgeConfounded_Rotated_Native
```
For non graph models e.g. Spherical UNet use:
 ```
 python train.py @params/SphericalUNet_ScanAge_Rotated_Native
 ```

Graph Models are: Monet, ChebNet, GConvNet (with and without TopK Pooling) and all use *train_graph.py*. 
All other models are non-graph models are executed with *train.py*.
Results will be automatically saved in the appropriate results folder.

# UGSCNN / Segemntation Experiments - Setting Experiment Parameters

To run a segmentation or UGSCNN regression experiment, go to the relevant folder within the Segmentation_UGSCNN directory. 
The appropriate experimental parameters must be changed manually. 

Within the dataloader, the unwarped and warped file directories must be specified manually (specifying either the native or template data paths depending on which experiment is being run).

The dataloaders are found in the following locations:
- for projected resnet: Projected_ResNet/MyDataLoader.py
- for Spherical UNet: Spherical_UNet/Spherical_UNet_Dataloader.py 
- for UGSCNN: meshcnn/utils.py
- for all Graph Models (MoNet, ChebNet, GConvNet): GraphMethods/graph_dataloader.py 

After setting the correct data locations, each experiment can be run from its corresponding executable python file.

Training Rotations must be manually set to either True or False from the train loader, depending on the choice of experiment:

e.g
```
train_loader = My_Projected_dHCP_Data_Segmentation(train_set, number_of_warps = 99, rotations= <#CHANGE HERE#>, smoothing = False, normalisation='std', parity_choice='both', output_as_torch=True)
```

# Loading Graph Models
For the Graph Methods, two separate executables exist: *segmentation.py* and *segmentation_TopK.py* for experiments without and with TopK models respectively.
Within these, the correct model (MoNet, ChebNet or GConvNet) must be loaded manually.

In *segmentation.py* load the following:
```
# For Monet:
model = monet_segmentation(num_features=[32,64,128,256,512,1024])

# For ChebNet:
model = GraphUNet_modded(conv_style=ChebConv,activation_function=nn.ReLU(), in_channels = 4, device=device)
    
# For GConvNet:
model = GraphUNet_modded(conv_style=GCNConv,activation_function=nn.ReLU(), in_channels = 4, device=device)
```

In *segmentation_TopK.py* each model can be loaded separately:

```
# FOR GConvNet With Top K:

model = GraphUNet_TopK_GCN(4,37,4,0.5,False,act=F.relu)

# FOR ChebNet With Top K

# model = GraphUNet_TopK_Cheb(4,37,4,0.5,False,act=F.relu)    
    
```

# Saving models and results

To save the coresponding models and results, the save directories must be manually changed.
