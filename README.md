# SwissPV
Swiss houseshold-level solar panel identification from ortho-rectified satellite images with deep learning. Using the [Inception-v3](https://arxiv.org/pdf/1512.00567.pdf) framework, we performed transfer learning with the pre-trained weights from US images as a part of the [DeepSolar](http://web.stanford.edu/group/deepsolar/home) project.
The model was developed using [Keras](https://keras.io/), on Python 3.6.5.

## Setting up the environment
The necessary librairies are explicited in `requirements.txt`. To install them all at once, run
```
pip3 install -r requirements.txt
```

## Running the program
### Classification
The script ```classification.py``` handles all aspects of the classification task (loads data, trains and evaluates model, saves checkpoints). Have a look at the possible flags for specific use. The saved Keras model is called ```keras_swisspv_XXX.h5```.

#### Flags

`--ckpt_load` Load the model from a saved .h5 file. [str]
`--ckpt_load_weights` Load the weights by giving the path with respect to the base directory. [str]
`--epochs` Number of training iterations. [int]
`--verbose` Print out information gradually in terminal output. [bool]
`--epochs_ckpt` Number of training iterations after which the model is saved. [int]
`--batch size` Batch size. [int]
`--train_set` Pickle file where distribution of train images is stored. If these files exist already, the program will load from them. [str]
`--test_set` Pickle file where distribution of test images is stored. If these files exist already, the program will load from them. [str]
`--validation_split` Fraction of images used for validation. [float]
`--skip_train` Skip training. [bool]
`--skip_train` Skip testing. [bool]
`--data_dir` Path to the dataset. Folder must contain two sub-folders "PV" and "noPV" containing the images with PV and withouth PV, respectively. [str]

**Note**: the paths are to be given with respect to the base directory.

### Segmentation
The script ```segmentation.py``` handles all aspects of the classification task (loads data, trains and evaluates model, saves checkpoints). Have a look at the possible flags for specific use.

#### Flags

`--ckpt_load` Load the model from a saved .h5 file. [str]
`--ckpt_load_weights` Load the weights by giving the path with respect to the base directory. [str]
`--epochs` Number of training iterations. [int]
`--verbose` Print out information gradually in terminal output. [bool]
`--epochs_ckpt` Number of training iterations after which the model is saved. [int]
`--batch size` Batch size. [int]
`--train_set` Pickle file where distribution of train images is stored. If these files exist already, the program will load from them. [str]
`--test_set` Pickle file where distribution of test images is stored. If these files exist already, the program will load from them. [str]
`--validation_split` Fraction of images used for validation. [float]
`--skip_train` Skip training. [bool]
`--skip_train` Skip testing. [bool]
`--second_layer_from_ckpt` Load second layer from checkpoints. [bool]
`--two_layers` Use two layers. [bool]
`--seg_1_weights` Load layer 1 weights. [str]
`--seg_2_weights` Load layer 2 weights. [str]
`--data_dir` Path to the dataset. Folder must contain two sub-folders "PV" and "noPV" containing the images with PV and withouth PV, respectively. [str]

**Note**: the paths are to be given with respect to the base directory.

## Authors & Acknowledgments

[JAKOB Anthony](https://github.com/antjak), [DHAENE Arnaud](https://github.com/arnauddhaene), and [ROMERO-GRASS Maëlle](https://github.com/maelleromero)

This is a project in the framework of the [Machine Learning CS-433 course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) given at EPFL (École Polytechnique Fédérale de Lausanne).

Project idea, data, and mentoring was provided by [Roberto Castello](https://people.epfl.ch/roberto.castello) from the [Solar Energy and Building Physics Laboratory LESO-PB](https://www.epfl.ch/labs/leso/).

A detailed description of the project, as well as a concise list of references, can be found in our report in the repository.
