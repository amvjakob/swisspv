# SwissPV
Swiss houseshold-level solar panel identification from ortho-rectified satellite images with deep learning. Using the [Inception-v3](https://arxiv.org/pdf/1512.00567.pdf) framework, we performed transfer learning with the pre-trained weights from US images as a part of the [DeepSolar](http://web.stanford.edu/group/deepsolar/home) project.
The model was developed using [Keras](https://keras.io/), on Python 3.6.5.

## Setting up the environment
The necessary librairies are explicited in `requirements.txt`. To install them all at once, run
```
pip install -r requirements.txt
```

Note: if your machine is unable to find Tensorflow 1.9.0, the code should run on any version of Tensorflow between 1.15.0 and 1.9.0.

## Data

The data is obtained from SwissTopo. Unfortunately, we cannot share them publicly.  

(for the CS-433 TA team) However, you can contact Dr. Roberto Castello of the LESO-PB lab if the data is absolutely necessary for grading the project.

## Running the program
### Final model

To start off, one needs to create the folders that will contain the models. To do this, run

```
mkdir ckpt
cd ckpt

mkdir deepsolar_classification
mkdir inception_tl_load
mkdir inception_tl_save
```

The final models can be downloaded [here](https://drive.google.com/drive/folders/1HpJn3-KUF0-MBD14KVLR7Xgo45swHH4E?usp=sharing),
and placed into `inception_tl_load`.

### Preprocessing
Alternatively, one can start off from scratch by downloading DeepSolar's model and creating the folders that will
contain the models. To do this, while in `ckpt`, run
``` 
curl -O https://s3-us-west-1.amazonaws.com/roofsolar/inception_classification.tar.gz
tar xzf inception_classification.tar.gz
```

We then have to transform DeepSolar's model from TensorFlow to Keras. To perform this task, return to the base directory and run

```
python deepsolar_to_tf.py
python tf_to_keras.py --with_aux=False
```
where `with_aux=False` creates a model with a single output, while `with_aux=True` uses an intermediary part of the model for predictions too and therefore produces two outputs. These two scripts take a total of approximately 20 minutes to run on a regular computer.

### Classification
The script ```train_classification_tl.py``` handles all aspects of the classification task (loads data, trains and evaluates model, saves checkpoints). Have a look at the possible flags for specific use. The saved Keras model is called ```keras_swisspv_untrained.h5``` and is found in ```/BASE_DIR/ckpt/inception_tl_load/```.
An example is provided in `run.sh`

Alternatively, one can use the file ```train_classification_tl_aux.py``` to run the classification using the augmented model.
An example is provided in `run_aux.sh`

#### Flags

`--ckpt_load` Load the given model from a saved .h5 file. [str]  
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

### Segmentation
The script ```train_segmentation_tl.py``` handles all aspects of the segmentation task (loads data, trains and evaluates model, saves checkpoints). Have a look at the possible flags for specific use.
An example is provided in `run_seg.sh`

#### Flags

`--ckpt_load` Load the given model from a saved .h5 file. [str]  
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

## Authors & Acknowledgments

[JAKOB Anthony](https://github.com/antjak), [DHAENE Arnaud](https://github.com/arnauddhaene), and [ROMERO-GRASS Maëlle](https://github.com/maelleromero)

This is a project in the framework of the [Machine Learning CS-433 course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) given at EPFL (École Polytechnique Fédérale de Lausanne).

Project idea, data, and mentoring was provided by [Roberto Castello](https://people.epfl.ch/roberto.castello) from the [Solar Energy and Building Physics Laboratory LESO-PB](https://www.epfl.ch/labs/leso/).

A detailed description of the project, as well as a concise list of references, can be found in our report in the repository.
