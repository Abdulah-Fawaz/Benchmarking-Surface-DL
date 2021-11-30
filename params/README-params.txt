Experimental Parameters are defined by a number of arguments, many of which are set as default. 

The parameters are:

- model: this is the model to be used. Valid options include: 'presnet, chebnet, chebnet_nopool, gconvnet, gconvnet_nopool, sphericalunet, monet_polar, and s2cnn_small'

chebnet and gconvnet refer to the topk pooling variations in the paper. chebnet_nopool and gconvnet_nopool are the non-pooling variants.
s2cnn_small and monet_polar are the architectures used in the paper.

-project: this is a boolean defining whether projection of the data to 2D is necessary. This will be False for all models except presnet and s2cnn_small

-task: this will be either 'regression' or 'regression_confounded' for scan age or confounded birth age experiments respectively.

-in_channels: this will always be set to 4 as there are 4 input channels

-dataset_arr: This defines which dataset to use. 'scan age' uses the scan age dataset and 'birth_age_confounded' uses the birth age dataset 

-patience: an int - usually set to 100 - defines the number of epochs the network will wait to obtain a better validation error before considering
 convergence as having occured and the experiment ending 

-train_rotations: a boolean that determines whether rotations should be used during training

-train_warps: an int that determines how many warps to use. usually set to 100 in the case that warps are desired or 0 if not.

-learning_rate: a float that determines the learning rate of the network.

-unwarp_dir: the location of the folder containing all of the unwarped datafiles
-warp_dir: the location of the folder containing all of the warped datafiles

The above two will change depending on whether native or template data will be used and must point to the saved data file directory


-features: the size of the network
for S2CNN, the features must be also set to 16,32,64,128,256

-train_bsize: the batch size during training. 

for monet, the batch size must be set to 1: