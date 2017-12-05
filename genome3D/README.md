Genome3D

Genome3D is a suite of predictive models for three-dimansional chromosome conformation. Genome3D predicts the interaction between two genomic loci in 3D nuclear space. Genome3D takes either of the two inputs: 
1) linear genomic features
2) sequence features. 
The models have been trained and tested using promoter-capture HiC data. 
You can read about data curation and model details in the manuscript: 

Dependencies

Genome3D is written in Python 2.7 (Anaconda Distribution 64-bit 2.7 for Linux: https://www.continuum.io/downloads) 
Following dependencies are required to run Genome3D:
- theano
- keras 
- sklearn
- numpy
Also cuda and cudnn need to be installed as well.

Installation

Genome3D can be installed with the following command from the terminal:

git clone https://github
cd genome3D
python setup.py install

Running Genome3D.

Genome3D include a set of predictive models: 
- Random Forest
- Linear Kernel Support Vector Machines (SVM)
- RBF Kernel SVM
- Fully Connected Deep Neural Network
- Siamese Deep Neural Network
- Deep Residual Networks 

Models can be imported from Genome3D:
from Genome3D.models_3d import Genome3D_RandomForest, Genome3D_SVM_Linear, Genome3D_SVM_RBF,  Genome3D_DNN_FC, Genome3D_DNN_Siamese, Genome3D_DNN_FC_ResNet

Every model object has train and predict attributes. For example, to train Ranfom Forest model:

rf = Genome3D_RandomForest(param)
rf.train(X_train, y_train)

To make predictions using Ranfom Forest model:

preds_test = rf.predict(X_test)

Hyper-parameter estimation is implemented for Random Forest and SVM models. 
Feature selection is implemented for Random Forest and Fully Connected Deep Neural Network (method: DeepLift) models
