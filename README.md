# genome3D

genome3D is a suite of predictive models for three-dimansional chromosome conformation. genome3D predicts the interaction between two genomic loci in 3D nuclear space. genome3D takes either of the two inputs:
1) linear genomic features
2) sequence features.

The models have been trained and tested using promoter-capture HiC data.
You can read about data curation and model details in the manuscript:
[Predicting 3D genome](https://www.dropbox.com/s/pnslvq4zrjvfq1z/manuscript.pdf?dl=0)

## Dependencies

genome3D is written in Python 2.7 (Anaconda Distribution 64-bit 2.7 for Linux: https://www.continuum.io/downloads)

Following dependencies are required to run genome3D:
- theano
- keras
- sklearn
- numpy

Also cuda and cudnn need to be installed as well.

## Installing

genome3D can be installed with the following command from the terminal:

```
git clone https://github.com/mtaranov/genome3D
cd genome3D
python setup.py install
```
## Running genome3D

genome3D includes a suite of predictive models:
- Random Forest
- Linear Kernel Support Vector Machines (SVM)
- RBF Kernel SVM
- Fully Connected Deep Neural Network
- Siamese Deep Neural Network
- Deep Residual Networks

Models can be imported from genome3D:

```
from genome3D.models_3d import Genome3D_RandomForest, Genome3D_SVM_Linear, Genome3D_SVM_RBF,  Genome3D_DNN_FC, Genome3D_DNN_Siamese, Genome3D_DNN_FC_ResNet
```

Example data set can be found in exmpl_data folder:

```
x_train = np.load('exmpl_data/X_train.npy')
x_valid = np.load('exmpl_data/X_valid.npy')
x_test = np.load('exmpl_data/X_test.npy')
y_valid = np.load('exmpl_data/y_valid.npy')
y_train = np.load('exmpl_data/y_train.npy')
y_test = np.load('exmpl_data/y_test.npy')
```
Every model has train and predict  attributes. For example, to train Fully Connected Deep Neural Network model:
```
dnn = Genome3D_DNN_FC(num_features=18)
dnn.train(x_train, y_train, (x_valid, y_valid))
```
To make predictions using DNN-FC model:
```
preds = dnn.predict(x_test)
```
To print out model performance:
```
print(dnn.test(x_test, y_test))
```

More examples can be found in the examples folder.

## Hyper-parameters

Hyper-parameter estimation is implemented for Random Forest and SVM models.

## Feature Importance

Feature selection is implemented for Random Forest and Fully Connected Deep Neural Network (method: DeepLift).

