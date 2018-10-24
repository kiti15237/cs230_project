import argparse
import os
import numpy as np
import os
import tempfile
import keras
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.callbacks import History 
from keras.models import load_model
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.layers import Concatenate
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import random
from termcolor import colored
from IPython.display import clear_output
from load_5hot import load
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type = str, help = "Datapath for input data, format blast output tab delimited qid  ref_id  qseq  refseq  eval")
parser.add_argument('--out_file', type = str, help = "output file path")
parser.add_argument('--logs_dir', type = str)
parser.add_argument('--weights_dir', type = str)
parser.add_argument('--plot_dir', type = str)
args = parser.parse_args()

input_filepath = args.in_file
logs_dir = args.logs_dir
plot_dir = args.plot_dir
weights_dir = args.weigths_dir

#### Load Data ####
[X_train, Y_train, target_strains_train, X_test, Y_test, target_strains_test, X_val,Y_val, target_strains_val] = load(input_filepath) #custom load function found in load_5hot.py

m = X_train.shape[0]
print("There are " + str(m) + " training examples")
print("There are " + str(X_test.shape[0]) + " testing examples")
print("There are " + str(X_train.shape[1]) + " classes: A, C, G, T, -")
max_length = max(X_train.shape[2],X_test.shape[2])
min_length = min(X_train.shape[2], X_test.shape[2])
print("The longest sequence is " + str(max_length) + " nucleotides long")
print("The shortest sequence is " + str(min_length) + " nucleotides long")
print("Should be the same due to padding")
print("X_train shape is:")
print(X_train.shape)

print('Permuting...')
np.random.seed(0)
rand_perm = np.random.rand(m).argsort()
np.take(X_train, rand_perm, axis=0, out=X_train)
print("finished X")
np.take(Y_train, rand_perm, axis=0, out=Y_train)
print("finished Y")
np.take

#Change dimensions so that we have examples x seqlen x onehot
X_train = np.squeeze(np.swapaxes(X_train, 1, 2))
Y_train = np.squeeze(np.swapaxes(Y_train, 1, 2))
X_val = np.squeeze(np.swapaxes(X_val, 1, 2))
Y_val = np.squeeze(np.swapaxes(Y_val, 1, 2))
X_test = np.squeeze(np.swapaxes(X_test, 1, 2))
Y_test = np.squeeze(np.swapaxes(Y_test, 1, 2))




####  Declare Model ####
one_hot_input = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
num_encoder_tokens = len(one_hot_input)
num_decoder_tokens = len(one_hot_input)
latent_dim = 100
encoder_inputs = Input(shape=(None, num_encoder_tokens))
e_lstm_1 = Bidirectional(LSTM(latent_dim, return_sequences = True))(encoder_inputs)
e_dropout = Dropout(0.5)(e_lstm_1)
e_lstm_2 = Bidirectional(LSTM(latent_dim, return_sequences = True))(e_dropout)
output = TimeDistributed(Dense(num_decoder_tokens, activation = "softmax"))(e_lstm_2)
model = Model(encoder_inputs, output)



#### Plot training time loss curves  ####
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.xlabel = "epochs"
        plt.ylabel = "loss"
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(plot_dir + "/training_loss_curve.pdf)
        
        self.losses_file = open('balanced_complete_logs/log_loss.txt', 'a')
        self.losses_file.write(str(logs.get('loss')) + ",")
        self.losses_file.close()
        
        self.val_loss_file = open('balanced_complete_logs/log_val_loss.txt', 'a')
        self.val_loss_file.write(str(logs.get('val_loss')) + ",")
        self.val_loss_file.close()
        
        self.val_acc_file = open('balanced_complete_logs/log_val_acc.txt', 'a')
        self.val_acc_file.write(str(logs.get('val_acc')) + ",")
        self.val_acc_file.close()
        
plot_losses = PlotLosses()
                    
####  Train Model  ####
adam = keras.optimizers.Adam(lr = .0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.95, amsgrad=False)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics = ['accuracy'])

history = History()
filepath= weights_dir + "/LSTM-2-comboseqs-dropout0.5-manytomany-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only= False, mode='max')

numExamples = 10000
batch_size = 100
epochs = 30

model.fit(x = X_train[0:numExamples], y = Y_train[0:numExamples],
          validation_data = (X_val, Y_val),
          batch_size=batch_size,
          epochs=epochs,
          verbose = 1,
          callbacks = [history, checkpoint, plot_losses])