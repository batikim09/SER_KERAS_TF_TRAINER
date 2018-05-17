# SER_KERAS_TF_TRAINER
This repository includes source codes and documents for Keras/Tensorflow based speech emotion recognition model (https://github.com/batikim09/LIVE_SER) training.

Maintainer: [**batikim09**](https://github.com/**github-user**/) (**batikim09**) - **j.kim@utwente.nl**

<a id="top"/>

This folder has source codes of a model trainer for speech emotion recognition. 

##Contents
1. <a href="#1--installation-requirements">Installation Requirements</a>

2. <a href="#2--usage">Usage</a>

3. <a href="#3--references">References</a>

## 1. Installation Requirements <a id="1--installation-requirements"/>
This software only runs on OSX or Linux (tested on Ubuntu). It is compatible with python 2.x and 3.x, but the following descrptions assume that python 3.x is installed.

### basic system packages

This software relies on several system packages that must be installed using a software manager.

For Ubuntu, please run the following steps:

`sudo apt-get install python-pip python-dev libhdf5-dev portaudio19-dev'

### python packages
Using pip, install all pre-required modules.
(pip version >= 8.1 is required, see: http://askubuntu.com/questions/712339/how-to-upgrade-pip-to-latest)

sudo pip3 install -r requirements.txt

## 2. Usage <a id="2--usage"/>

We assume that users already downloaded the eNTERFACE corpus that is freely available (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.220.2113&rep=rep1&type=pdf) and builded a h5 database. See details in feature extractor (https://github.com/batikim09/SER_FEAT_EXT).

We assume the location of the database is "../SER_FEAT_EXT/h5db/ENT.RAW.3cls.av".

With this small corpus, deep temporal architectures can't provide any benefits. The following scripts show just examples that do not have any fine-tunning.

### Basic training
Users can combine various types of neural networks such as fully-connected neural network (FCN), convolutional neural network (CNN), long-short-term-memory (LSTM), residual network (RESNET), and highway. 

Feature vectors have temporal structures. For example, the 2D feature input has a shape of (#sample, #time, 1, #context_window, #feature_dim). The 3D feature input has a shape of (#sample, 1, #time, #context_window, #feature_dim). See details of context windows in https://github.com/batikim09/. See "./scripts/basic.sh".

### Updating pretrained models

Users can train a background model first and load it for fine-tunning. When re-updating parameters of a pre-trained model, freezing some layers is possible too. See "./scripts/pretrained.sh".

### Balanced learning
To deal with imbalanced distributions of classes, several methods are provided. See "./scripts/balanced_learning.sh".

## 3. References <a id="3--references"/>

This software is based on the following papers. Please cite one of these papers in your publications if it helps your research:

@inproceedings{kim2017interspeech,
  title={Towards Speech Emotion Recognition ``in the wild'' using Aggregated Corpora and Deep Multi-Task Learning},
  author={\textbf{Kim, Jaebok} and Englebienne, Gwenn and Truong, Khiet P and Evers, Vanessa},
  booktitle={Proceedings of the INTERSPEECH},
  pages={1113--1117},
  year={2017}
}


@inproceedings{kim2017acmmm, title={Deep Temporal Models using Identity Skip-Connections for Speech Emotion Recognition}, author={Kim, Jaebok and Englebienne, Gwenn and Truong, Khiet P and Evers, Vanessa}, booktitle={Proceedings of ACM Multimedia}, pages={1006-1013}, year={2017} }

@inproceedings{kim2017acii, title={Learning spectro-temporal features with 3D CNNs for speech emotion recognition}, author={Kim, Jaebok and Truong, Khiet and Englebienne, Gwenn and Evers, Vanessa}, booktitle={Proceedings of International Conference on Affective Computing and Intelligent Interaction}, pages={}, year={2017} }

<a id="top"/> 
