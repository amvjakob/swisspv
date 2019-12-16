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
The script ```classification.py``` handles all aspects of the classification task (loads data, trains and evaluates model, saves checkpoints). Have a look at the possible flags for specific use, and be careful to specify the location of the data under IMG_DIR. The saved Keras model is called ```keras_swisspv.h5```.

### Segmentation
The script ```segmentation.py``` handles all aspects of the classification task (loads data, trains and evaluates model, saves checkpoints). Have a look at the possible flags for specific use. The saved Keras model is called ```keras_swisspv.h5```.

## Authors

[JAKOB Anthony](https://github.com/antjak), [DHAENE Arnaud](https://github.com/arnauddhaene), and [ROMERO-GRASS Maëlle](https://github.com/maelleromero)
