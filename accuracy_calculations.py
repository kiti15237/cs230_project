from onehot_tostring import convert_to_nucs
import numpy as np

def acc_sequences_correct(y_predicted, y_true):
    # Takes in two arrays of shape (n,l,5)
    noiseless = 0
    num_examples = y_predicted.shape[0]
    for n in range(num_examples):
        samp_n_pred = convert_to_nucs(np.swapaxes(y_predicted[n,:,:], 0, 1))
        samp_n_true = convert_to_nucs(np.swapaxes(y_true[n,:,:], 0, 1))
        
        if samp_n_pred == samp_n_true:
            noiseless = noiseless + 1
            
    return noiseless/num_examples