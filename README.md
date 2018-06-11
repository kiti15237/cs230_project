# cs230_project

Contributors:
1. Christine Tataru
2. Clara McCreery
3. Abhishek Roushan

Data:
Study PRJEB6244

Link: http://www.ebi.ac.uk/ena/data/view/PRJEB6244

Note: All data are stored local to our machines and shared via google drive

The project attempts to tackle the DNA sequence denoising from two approaches:
1. Convolutional Approach (CNN)
2. Recurrent Approach (RNN)

Organization:

The data loading and processing codes are available in the following files:
  - load_data.py: general loading data from .out file into 4D one hot vector and prepare train/test data 
  - load_5hot.py : loading dat from .out file into 5D one-hot vector and prepare train/val/test data
  

The convolutional neural network related codes are listed in the following files:
  - ConvNet_5hot.ipynb : working notebook for conv-net model with 5D one-hot input vectors
  - ConvNet_customLoss.ipynb : attempt for custom loss function (loss by sequence) in conv-net model
  - convWeights/ : directory which stores all conv-net weights during training phase. User can use the weights to load model.
 
The recurrent neural network related codes are listed in the following files:
  - seq2seq_working.ipynb : Working notebook for sequential RNN model with 5D one-hot input vector
  - seq2seqMultiLoss.ipynb : attempt for hybrid loss (noise classification [y/n] and total sequence prediction); work in progress
  - seqWeights/ : directory which stores all sequential model weights during training phase. Use weights to load model.
  
Evaluation (accuracy/prediction labels) codes are listed in the following files:
  - accuracy_calculations.py: accuracy of prediction and confusion matrix code
  - onehot_tostring.py : visualization of predictions; hardmax, noise display code.
  
Miscillaneous:
  - NoiseClassification.ipynb : Noise classification [y/n] of input sequences CONV-NET MODEL working notebook; Work in progress
  - Seq2SeqNoise.ipynb: Noise classification [y/n] of input sequences RNN-MODEL working notebook; Work in progress.
  

Work by running the ipython notebooks or simply running .py files. Enjoy!

  
