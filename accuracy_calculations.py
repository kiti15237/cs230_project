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
    ind_to_char = np.array(['A', 'C','G', 'T', '-'])
    char_to_ind = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
    if verbose:
        print(seq_lengths[0:10])
    baseline_error = 0
    pred_error = 0
    bad_changes = 0
    fail_to_change = 0
    good_changes = 0
    
    conf_mat = {}
    conf_mat['good'] = np.zeros((5, 5))
    conf_mat['bad'] = np.zeros((5, 5))
    conf_mat['fail'] = np.zeros((5, 5))
    conf_mat['baseline'] = np.zeros((5,5))
    
    for i in range(y_predicted.shape[0]):
        target_char = ind_to_char[np.argmax(y_true[i,0:seq_lengths[i]], axis = -1)]
        test_char = ind_to_char[np.argmax(X_data[i,0:seq_lengths[i]], axis = -1)]
        pred_char = ind_to_char[np.argmax(y_predicted[i,0:seq_lengths[i]], axis = -1)]

        baseline_error += np.sum(target_char != test_char) 
        pred_error += np.sum(target_char != pred_char) 

        to_red = np.where((target_char != pred_char) & (target_char == test_char))[0]
        to_green = np.where((target_char != test_char) & (target_char == pred_char))[0]
        to_blue = np.where((target_char != test_char) & (target_char != pred_char))[0]
        should_change = np.where((target_char != test_char))[0]
        
        for ind in to_green:
            ind1 = char_to_ind[test_char[ind]]
            ind2 = char_to_ind[pred_char[ind]]
            conf_mat['good'][ind1, ind2] += 1
        for ind in to_red:
            ind1 = char_to_ind[test_char[ind]]
            ind2 = char_to_ind[pred_char[ind]]
            conf_mat['bad'][ind1, ind2] += 1
        for ind in to_blue:
            ind1 = char_to_ind[test_char[ind]]
            ind2 = char_to_ind[target_char[ind]]
            conf_mat['fail'][ind1, ind2] += 1
        for ind in should_change:
            ind1 = char_to_ind[test_char[ind]]
            ind2 = char_to_ind[pred_char[ind]]
            conf_mat['baseline'][ind1, ind2] += 1

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
    errors = {'conf_mat':conf_mat, 'perSeqErr': pred_error/norm_seq, 'perBaseErr': pred_error/norm_base}              
    
    return(errors)


#Usage:
#1. errors = calcError(y_predicted , y_true , X_data = , verbose = True)
#2. plot_conf(errors)
def plot_ind_conf(fig, mat, type_pred, i):
    ax = fig.add_subplot(2, 2, i)
    cax = ax.matshow(mat)
    ax.tick_params(labelsize= 20) 
    cb = fig.colorbar(cax)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize= 20)
    ax.set_xticklabels([''] + ['A', 'C','G', 'T', '-'])
    ax.set_yticklabels([''] + ['A', 'C','G', 'T', '-'])
    plt.ylabel("From", fontsize = 20)
    plt.title( type_pred + " predictions \n \n To \n", fontsize = 20)
    
def plot_conf(errors):
    fig = plt.figure(figsize = (15,15))  
    fig.subplots_adjust(hspace= 0.4, wspace=0.4)
    plot_ind_conf(fig, errors['conf_mat']['good'], "Good", 1)
    plot_ind_conf(fig, errors['conf_mat']['bad'], "Bad", 2)
    plot_ind_conf(fig, errors['conf_mat']['fail'], "Fail To Change", 3)
    plot_ind_conf(fig, errors['conf_mat']['baseline'], "Baseline", 4)

