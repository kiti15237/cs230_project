
# coding: utf-8

# In[122]:


#imports
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
from keras.layers import BatchNormalization
from keras.layers import Conv2D, Add, Reshape
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
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
import datetime
from termcolor import colored

from keras.activations import softmax 
print("import loads done")


# In[38]:


def model_1dconv(window_size, filter_len, n_inp_series=1, n_output=1, n_filter=4):
    model= Sequential()
    model.add(Conv1D(filters= n_filter, kernel_size= filter_len, activation="relu", input_shape=(window_size, n_inp_series)))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=n_filter, kernel_size=filter_len, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(n_output, activation="linear"))
    
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


# In[40]:


#run the cell if want to progress using 1d conv model
model= model_1dconv(window_size=50, filter_len= 3, n_inp_series=5, n_output=5, n_filter=4)
model.summary()


# In[98]:


datapath= "blast_tab_1hit.out"

file= open(datapath, 'r')

train_entries=[]
val_entries= []
test_entries=[]
max_length_in =0
max_length_out=0
ynoise_train=[]
ynoise_val=[]
ynoise_test=[]

np.random.seed(0)

for ln in file:
    toks=ln.split('\t')
    rand_num= np.random.random()
    
    if(toks[2] != toks[3]):
        max_length_in= max(max_length_in, len(toks[2]))
        max_length_out= max(max_length_out, len(toks[3]))
        
        if rand_num< 0.95:
            train_entries.append([toks[2], toks[3]])
            ynoise_train.append([1])
        elif rand_num < 0.975:
            test_entries.append([toks[2], toks[3]])
            ynoise_test.append([1])
        else:
            val_entries.append([toks[2], toks[3]])
            ynoise_val.append([1])
    
    if(toks[2]== toks[3]):
        if rand_num > 0.975:
            val_entries.append([toks[2], toks[3]])
            ynoise_val.append([-1])
        elif rand_num > 0.95:
            test_entries.append([toks[2], toks[3]])
            ynoise_test.append([-1])
        elif rand_num > 0.9:
            train_entries.append([toks[2], toks[3]])
            ynoise_train.append([-1])
            
file.close()

label_train_noise= np.array(ynoise_train)
label_test_noise= np.array(ynoise_test)
label_val_noise= np.array(ynoise_val)

one_hot_input = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '-': 4}
one_hot_output = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
            


# In[99]:


#print shapes:
print("labels shape")
print("y_train: ", label_train_noise.shape)
print("y_test: ", label_test_noise.shape)
print("y_val: ", label_val_noise.shape)


# In[55]:


#visualize labels
num_vis= 100
#print("visualize labels for noisy vs non-noisy")
#print("y_train")
#print(label_train_noise[0:num_vis])
#print("y_val")
#print(label_val_noise[0:num_vis])
#print("y_test")
#print(label_test_noise[0:num_vis])


# In[57]:


#visualize data entries
n_peek=10
print("data entries:")
print("train")
print(train_entries[n_peek])

print("test")
print(test_entries[n_peek])

print("val")
print(val_entries[n_peek])


# In[100]:


#length of sequences
print("max len inp:", max_length_in)
print("max len out: ", max_length_out)


# In[101]:


labels_train=np.array(label_train_noise)
labels_val= np.array(label_val_noise)
labels_test= np.array(label_test_noise)

print("shapes")
print(labels_train.shape)
print(labels_test.shape)
print(labels_val.shape)


# In[102]:


#The next few cells used for data preparation
#for train, val and test data to send inside model

input_seqs= [entry[0] for entry in train_entries]
output_seqs= [entry[1] for entry in train_entries]
val_input_seqs= [entry[0] for entry in val_entries]
val_output_seqs= [entry[1] for entry in val_entries]
test_input_seqs= [entry[0] for entry in test_entries]
test_output_seqs= [entry[1] for entry in test_entries]


# In[103]:


print(len(input_seqs))
print(len(val_input_seqs))
print(len(test_input_seqs))


# In[104]:


#train val and test data
X_train= np.zeros(
    (len(input_seqs), len(one_hot_input), max_length_in, 1),
    dtype= 'float32')
#print(X_train.shape)
for i, (input_seqs, output_seqs) in enumerate(zip(input_seqs, output_seqs)):
    for t, char in enumerate(input_seqs):
        X_train[i,one_hot_input[char],t,0]=1


# In[105]:


#shape and data for X_train
print(X_train.shape)
#print(X_train[0:num_vis, :,:,0])


# In[106]:


#val data train
X_val= np.zeros(
    (len(val_input_seqs), len(one_hot_input), max_length_in, 1),
    dtype= 'float32')
#print(X_train.shape)
for i, (val_input_seqs, val_output_seqs) in enumerate(zip(val_input_seqs, val_output_seqs)):
    for t, char in enumerate(val_input_seqs):
        X_val[i,one_hot_input[char],t,0]=1


# In[107]:


#shape and data for X_val
print(X_val.shape)
#print(X_val[0:num_vis, :,:,0])


# In[108]:


X_test= np.zeros(
    (len(test_input_seqs), len(one_hot_input), max_length_in, 1),
    dtype= 'float32')
#print(X_train.shape)
for i, (test_input_seqs, test_output_seqs) in enumerate(zip(test_input_seqs, test_output_seqs)):
    for t, char in enumerate(test_input_seqs):
        X_test[i,one_hot_input[char],t,0]=1


# In[109]:


#shape and data for X_test
print(X_test.shape)
#print(X_test[0:num_vis, :,:,0])


# In[110]:


def softMaxAxis1(x):
    return softmax(x,axis=1)


def model_custom(input_shape):
    X_input= Input(input_shape)
    
    X= Conv2D(128, (4,7), strides=(1,1), padding='same', name='conv0')(X_input)
    X= BatchNormalization(axis=3, name='bn0')(X)
    X= Activation('relu')(X)
    
    X= Conv2D(64, (1,7), strides=(1,1), padding='same', name='conv1')(X)
    X= BatchNormalization(axis=3, name='bn1')(X)
    X= Activation('relu')(X)
    
    X= Conv2D(64, (1,7), strides=(1,1), padding='same', name='conv2')(X)
    X= BatchNormalization(axis=3, name='bn2')(X)
    X= Activation('relu')(X)
    
    X= Dropout(0.3, name='dropout')(X)
    
    X= Conv2D(32, (1,7), strides=(1,1), padding='same', name='conv3')(X)
    X= BatchNormalization(axis=3, name='bn3')(X)
    X= Activation('relu')(X)
    
    X= Conv2D(1, (1,1), strides=(1,1), padding='same', name='conv4')(X)
    X= BatchNormalization(axis=3, name='bn4')(X)
    X= Flatten()(X)
    X= Dense(1, activation='sigmoid')(X)
    
    model= Model(inputs=X_input, outputs=X, name='Model1')
    return model
    


# In[111]:


#need to pass in parameters from input data
mymodel= model_custom((len(one_hot_input),max_length_in,1))
mymodel.summary()


# In[112]:


#compile model
adam= keras.optimizers.Adam(lr= 0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#mymodel.compile(optimizer="Adam", loss= "binary_crossentropy", metrics=["accuracy"])
mymodel.compile(optimizer=adam, loss= "binary_crossentropy", metrics=["accuracy"])
print("compilation done!")


# In[118]:


#run model
history = History()
filepath="seqWeights/ConvNet-test-adam_lr1e-2-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
numExamples = 1000
batch_size = 128
epochs = 20
print(X_train.shape)
print((X_train[0:numExamples,:,:,:]).shape)
#mymodel.fit(X_train[0:numExamples, :, :,:],
#          np.array(labels_train)[0:numExamples],
#          batch_size=batch_size,
#          epochs=epochs,
#          validation_data=(X_val[0:numExamples,:,:,:], np.array(labels_val)[0:numExamples]), verbose = 1,
#         callbacks = [history, checkpoint])
mymodel.fit(X_train,
          np.array(labels_train),
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_val, np.array(labels_val)), verbose = 1,
         callbacks = [history, checkpoint])


# In[120]:


np.array(labels_val).shape


# In[123]:


currtime = datetime.datetime.now()
histfname = "./trainHistoryDict_" + currtime.strftime("%m%d-%H%M") + "_5hotNoiseClassification"
with open(histfname, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


# In[124]:


histfname = "./trainHistoryDict_" + currtime.strftime("%m%d-%H%M") + "_5hotNoiseClassification"
history = pickle.load(open(histfname, "rb" ))


# In[126]:


plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()

