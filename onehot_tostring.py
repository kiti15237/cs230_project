import numpy as np

def convert_to_nucs(one_hot):
    # Takes in a one-hot vector of shape (5, length)
    rows = np.argmax(one_hot,axis=0)
    conf = np.max(one_hot,axis=0)
    nucs = ''
    for idx in range(one_hot.shape[1]):
        if(conf[idx]>0.3):
            nuc = rows[idx]
        else:
            nuc = '-'
            
        if(nuc == 0):
            nucs = nucs+'A'
        elif(nuc == 1):
            nucs = nucs+'C'
        elif(nuc == 2):
            nucs = nucs+'G'
        elif(nuc == 3):
            nucs = nucs+'T'
        else:
            nucs = nucs+'-'
            
    return nucs

def convert_to_hardmax(one_hot):
    # Takes in a one-hot vector of shape (5, length)
    rows = np.argmax(one_hot,axis=0)
    conf = np.max(one_hot,axis=0)
    for idx in range(one_hot.shape[1]):
        one_hot[:,idx] = 0
        if(conf[idx]>0.3):
            one_hot[rows[idx],idx] = 1
            
    return one_hot

def show_noise(pred,denoised,noisy):
    noise = ''
    for idx in range(len(pred)):
        if(pred[idx] == denoised[idx] and pred[idx] == noisy[idx]):
            noise = noise+'-'
        elif(pred[idx] == denoised[idx]):
            noise = noise+'g'
        else:
            noise = noise+'b'
    return noise

