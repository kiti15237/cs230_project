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


def calcError(y_predicted, y_true, X_data, verbose = False):
   #Returns # of noisy nucleotides when comparing the predicted seqs and the target seqs perBase or perSequence. Error perBase should be interpreted as the probability that any given base output by the model is incorrect. Error perSequence should be interpreted as the average number of incorrect output bases per sequence. So an error of .03 means there at .03 incorrect bases in every sequence. This translates to 3 incorrect bases every 100 sequences. 

    seq_lengths = [np.argmin(np.sum(X_data[i,:,:], axis = 1)) for i in range(X_data.shape[0])] #assumes 0000 for end of sequence. If there is no 0000, this will mess up, but I don't think that happens
    output_dict = np.array(['A', 'C','G', 'T', '-'])
    if verbose:
        print(seq_lengths[0:10])
    baseline_error = 0
    pred_error = 0
    bad_changes = 0
    fail_to_change = 0
    good_changes = 0
    
    for i in range(y_predicted.shape[0]):
        target_char = output_dict[np.argmax(y_true[i,0:seq_lengths[i]], axis = -1)]
        test_char = output_dict[np.argmax(X_data[i,0:seq_lengths[i]], axis = -1)]
        pred_char = output_dict[np.argmax(y_predicted[i,0:seq_lengths[i]], axis = -1)]

        baseline_error += np.sum(target_char != test_char) 
        pred_error += np.sum(target_char != pred_char) 

        to_red = np.where((target_char != pred_char) & (target_char == test_char))[0]
        to_green = np.where((target_char != test_char) & (target_char == pred_char))[0]
        to_blue = np.where((target_char != test_char) & (target_char != pred_char))[0]

        bad_changes += len(to_red)
        good_changes += len(to_green)
        fail_to_change += len(to_blue)

        norm_seq = y_predicted.shape[0]
        norm_base = np.sum(seq_lengths)
    if verbose:
        print("Baseline error(num mismatched nucleotides between target and input average per sequence): " + str(baseline_error / norm_seq))
        print("Prediction error(num mismatched nucleotides between target and prediction average per sequence): " + str(pred_error / norm_seq))
        print( '\x1b[32m' + "Average good changes per sequence : " + '\x1b[0m'  + str(good_changes / norm_seq))
        print('\x1b[31m' + "Average bad changes per sequence: " + '\x1b[0m' + str(bad_changes / norm_seq))
        print('\x1b[34m' + "Average failure to change per sequence: "+ '\x1b[0m'  + str(fail_to_change / norm_seq)) 
    
    return(pred_error / norm_base, pred_error / norm_seq)


