
# coding: utf-8

# In[1]:


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
print("import loads done")


# In[2]:


# In[4]:


#data
# UPLOAD DATA
# (each user should put datafiles in this directory on their computer)
datapath = "blast_tab_1hit.out"
file = open(datapath, 'r')

same_entries = []
diff_entries = []
train_entries=[]
test_entries=[]
val_entries=[]
max_length_in = 0
max_length_out = 0
count_diff_entries_train=0
count_same_entries_train=0
count_diff_entries_val=0
count_same_entries_val=0
count_diff_entries_test=0
count_same_entries_test=0
y_train=[]
y_val=[]
y_test=[]

def plot_loss(history):
    epochs = len(history["loss"])
    plt.plot(range(0, epochs), history["loss"])
    plt.plot(range(0, epochs), history["val_loss"])
    plt.show()

np.random.seed(0)

for ln in file:
    toks = ln.split('\t')
    rand_num= np.random.random()
    
    max_length_in = max(max_length_in,len(toks[2]))
    max_length_out = max(max_length_out,len(toks[3]))
    if(toks[2] != toks[3]):
        if rand_num < 0.95:
            train_entries.append([toks[2], toks[3]])
            count_diff_entries_train += 1
            y_train.append([1])
        elif rand_num <0.975:
            test_entries.append([toks[2], toks[3]])
            count_diff_entries_test += 1
            y_test.append([1])
        else:
            val_entries.append([toks[2], toks[3]])
            count_diff_entries_val += 1
            y_val.append([1])
        
    if toks[2] == toks[3]:
        if rand_num > 0.975:
            val_entries.append([toks[2], toks[3]])
            count_same_entries_val += 1
            y_val.append([-1])
        elif rand_num > 0.95:
            test_entries.append([toks[2], toks[3]])
            count_same_entries_test += 1
            y_test.append([-1])
        elif rand_num > 0.9:
            train_entries.append([toks[2], toks[3]])
            count_same_entries_train += 1
            y_train.append([-1])
        #same_entries.append([toks[2], toks[3]])
    #else:
        #diff_entries.append([toks[2], toks[3]])

file.close()
#num_entries = len(same_entries) + len(diff_entries)

y_train_noise=np.array(y_train)
y_val_noise= np.array(y_val)
y_test_noise=np.array(y_test)
#display train, test, val set
#print("train")
#print(train_entries[2])
#print("val")
#print(val_entries[2])
#print("test")
#print(test_entries[2])

#diff_entries_input = [entry[0] for entry in diff_entries]
#diff_entries_output = [entry[1] for entry in diff_entries]
#same_entries_input = [entry[0] for entry in same_entries]
#same_entries_output = [entry[1] for entry in same_entries]
#diff_entries_output = [("\t" + entry[1] + "\n") for entry in diff_entries] #use '\t' as start character and '\n' as end character
#Visualize
#diff_entries_output[1]
one_hot_input = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '-': 4}
one_hot_output = {'A': 0, 'T': 1, 'C': 2, 'G': 3}


# In[5]:


print("y shapes")
print(y_train_noise.shape)
print(y_val_noise.shape)
print(y_test_noise.shape)


# In[6]:


#prints
print("train len",len(train_entries))
print("val len",len(val_entries))
print("tets len",len(test_entries))
print("diff entries in train", count_diff_entries_train)
print("same entries train", count_same_entries_train)
print("diff entries in val", count_diff_entries_val)
print("same entries val", count_same_entries_val)
print("diff entries in test", count_diff_entries_test)
print("same entries test", count_same_entries_test)

print("total train + dev + test")
print(len(train_entries)+ len(val_entries)+ len(test_entries))

print("max len in",max_length_in)
print("max len out", max_length_out)

print("one train entry",train_entries[1])


# In[7]:


input_seqs= [entry[0] for entry in train_entries]
output_seqs= [entry[1] for entry in train_entries]
val_input_seqs= [entry[0] for entry in val_entries]
val_output_seqs= [entry[1] for entry in val_entries]
test_input_seqs= [entry[0] for entry in test_entries]
test_output_seqs= [entry[1] for entry in test_entries]
#display
#print("train in")
#print(input_seqs[100])
#print("val in")
#print(val_input_seqs[100])
#print("test in")
#print(test_input_seqs[100])


# In[8]:


print(len(input_seqs))
print(len(output_seqs))
print(len(val_input_seqs))
print(len(test_input_seqs))


# In[9]:


labels_train= np.array(y_train_noise)
labels_val=   np.array(y_val_noise)
labels_test=  np.array(y_test_noise) 
#labels_train= [[1]]* count_diff_entries_train + [[-1]]*count_same_entries_train
#labels_val= [[1]]* count_diff_entries_val + [[-1]]*count_same_entries_val
#labels_test= [[1]]* count_diff_entries_test + [[-1]]*count_same_entries_test
print("train labels len",len(labels_train))
print("val labels len",len(labels_val))
print("test labels len",len(labels_test))


# In[11]:


c = list(zip(input_seqs, output_seqs, labels_train))
#seed = 123
#random.seed(seed)
#random.shuffle(c)
input_seqs, output_seqs, labels_train = zip(*c)
print("lenghts:")
print("input_seqs->",len(input_seqs))
print("output_seqs ->", len(output_seqs))
print("labels train-->", len(labels_train))


# In[12]:


c = list(zip(val_input_seqs, val_output_seqs, labels_val))
#seed = 123
#random.seed(seed)
#random.shuffle(c)
val_input_seqs, val_output_seqs, labels_val = zip(*c)
print("lenghts:")
print("input_seqs->",len(val_input_seqs))
print("output_seqs ->", len(val_output_seqs))
print("labels train-->", len(labels_val))


# In[13]:


c = list(zip(test_input_seqs, test_output_seqs, labels_test))
#seed = 123
#random.seed(seed)
#random.shuffle(c)
test_input_seqs, test_output_seqs, labels_test = zip(*c)
print("lenghts:")
print("input_seqs->",len(test_input_seqs))
print("output_seqs ->", len(test_output_seqs))
print("labels train-->", len(labels_test))


# In[4]:


#NOT TO RUN
#data massage
#numSameEntries = 50000
#input_seqs = diff_entries_input + same_entries_input[0:numSameEntries]
#output_seqs = diff_entries_output + same_entries_output[0:numSameEntries]
#labels = [[1]] * len(diff_entries_input) + [[-1]] * numSameEntries
#print(len(labels))
#print(len(input_seqs))


# In[105]:


print("arguments for below cell")
print("input seqs len: ", len(input_seqs))
#print("input seqs shape: ", input_seqs.shape)
print("max len in: ", max_length_in)
print("one hot inp len: ", len(one_hot_input))


# In[14]:


#one hot encode train data

#ENCODE
encoder_input_data = np.zeros(
    (len(input_seqs), max_length_in, len(one_hot_input)),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_seqs), max_length_out, len(one_hot_input)),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_seqs), max_length_out, len(one_hot_input)),
    dtype='float32')

for i, (input_seqs, output_seqs) in enumerate(zip(input_seqs, output_seqs)):
    for t, char in enumerate(input_seqs):
        #print(t)
        encoder_input_data[i, t, one_hot_input[char]] = 1.
    for t, char in enumerate(output_seqs):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, one_hot_input[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, one_hot_input[char]] = 1


# In[15]:


print(encoder_input_data.shape)
print(decoder_input_data.shape)


# In[16]:


#one hot encode val data

#ENCODE
encoder_input_data_val = np.zeros(
    (len(val_input_seqs), max_length_in, len(one_hot_input)),
    dtype='float32')
decoder_input_data_val = np.zeros(
    (len(val_input_seqs), max_length_out, len(one_hot_input)),
    dtype='float32')
decoder_target_data_val = np.zeros(
    (len(val_input_seqs), max_length_out, len(one_hot_input)),
    dtype='float32')

for i, (val_input_seqs, val_output_seqs) in enumerate(zip(val_input_seqs, val_output_seqs)):
    for t, char in enumerate(val_input_seqs):
        #print(t)
        encoder_input_data_val[i, t, one_hot_input[char]] = 1.
    for t, char in enumerate(val_output_seqs):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data_val[i, t, one_hot_input[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data_val[i, t - 1, one_hot_input[char]] = 1


# In[17]:


print(encoder_input_data_val.shape)
print(decoder_input_data_val.shape)


# In[18]:


#one hot encode test data

#ENCODE
encoder_input_data_test = np.zeros(
    (len(test_input_seqs), max_length_in, len(one_hot_input)),
    dtype='float32')
decoder_input_data_test = np.zeros(
    (len(test_input_seqs), max_length_out, len(one_hot_input)),
    dtype='float32')
decoder_target_data_test = np.zeros(
    (len(test_input_seqs), max_length_out, len(one_hot_input)),
    dtype='float32')

for i, (test_input_seqs, test_output_seqs) in enumerate(zip(test_input_seqs, test_output_seqs)):
    for t, char in enumerate(test_input_seqs):
        #print(t)
        encoder_input_data_test[i, t, one_hot_input[char]] = 1.
    for t, char in enumerate(test_output_seqs):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data_test[i, t, one_hot_input[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data_test[i, t - 1, one_hot_input[char]] = 1


# In[19]:


print(encoder_input_data_test.shape)
print(decoder_input_data_test.shape)


# In[20]:


#parameters to change
num_encoder_tokens = len(one_hot_input)
num_decoder_tokens = len(one_hot_input)
latent_dim = 100


# In[21]:


#model
encoder_inputs = Input(shape=(None, num_encoder_tokens))
e_lstm_1 = Bidirectional(LSTM(latent_dim, return_sequences = True))(encoder_inputs)
e_dropout = Dropout(0.5)(e_lstm_1)
e_lstm_2, fh2, fc2, bh2, bc2 = Bidirectional(LSTM(latent_dim, return_sequences = True, return_state=True))(e_dropout)
#e_lstm_3 = Bidirectional(LSTM(latent_dim, return_sequences = True))
h2=Concatenate()([fh2,bh2])
c2=Concatenate()([fc2,bc2])

#output = TimeDistributed(Dense(num_decoder_tokens, activation = "softmax"))(e_lstm_2)
noise_class = Dense(1, activation = "tanh")(Concatenate()([h2, c2]))
#noise_class = Activation("tanh")(noise_class)
#print(noise_class)


# In[22]:


model= Model(encoder_inputs, noise_class)
model.summary()

#np.array(labels)[0:2]


# In[23]:


#model = Model(encoder_inputs, noise_class)

#model.load_weights("seqWeights/LSTM-comboseqs-dropout0.5-manytomany-01-0.82.hdf5")
adam = keras.optimizers.Adam(lr = .001, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=0.95, amsgrad=False)
model.compile(optimizer= adam, loss='binary_crossentropy', metrics = ['accuracy'])

history = History()
filepath="seqWeights/LSTM-test-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

numExamples = 200
batch_size = 64
epochs = 40
#output_seqs = decoder_input_data[0:numExamples, :, :]
#y = output_seqs.reshape(numExamples, output_seqs.shape[1], 1)
#model.fit(encoder_input_data[0:numExamples, :, :],
#          np.array(labels)[0:numExamples],
#          batch_size=batch_size,
#          epochs=epochs,
#          validation_split=0.2, verbose = 1,
#         callbacks = [history, checkpoint])

model.fit(encoder_input_data,
          np.array(labels_train),
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(encoder_input_data_val, np.array(labels_val)), verbose = 1,
         callbacks = [history, checkpoint])


# In[117]:


#score= model.evaluate(encoder_input_data_test[0:numExamples,:,:], np.array(labels_test)[0:numExamples], verbose=0)
#
#
## In[118]:
#
#
#print("scores")
#print(score)
#
#
## In[24]:
#
#
print(model.predict(encoder_input_data_test))
print(np.array(labels_test))

lr = 0.001
save_dir= 'history/'
model.save(save_dir+'s2s_lr'+str(lr)+'_ep'+str(epochs)+'.h5')
with open(save_dir+'history_lr'+str(lr)+'_ep'+str(epochs), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

h = pickle.load(open(save_dir+'history_lr' + str(lr)+'_ep'+str(epochs), "rb"))
plot_loss(h)

