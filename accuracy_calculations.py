from onehot_tostring import convert_to_nucs, show_noise

def calc_sequences_correct(y_predicted, y_true):
    noiseless = 0
    num_examples = y_predicted.shape[0]
    for n in range(num_examples):
        samp_n_pred = convert_to_nucs(y_predicted[n,:,:,0])
        samp_n_true = convert_to_nucs(y_true[n,:,:,0])
        
        if samp_n_pred == samp_n_true:
            noiseless = noiseless + 1
            
    return noiseless/num_examples