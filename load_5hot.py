import numpy as np

def one_hot_sequence(s):
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
    s_num = [char_to_int[c] for c in s]
    n_labels = 5
    one_hot = np.eye(n_labels)[:,s_num]
    return(one_hot) #convert to one-hot column-wise


def one_hot_list(entries,max_length):
    m = len(entries)
    one_hot_ins = np.zeros((m,5,max_length))
    one_hot_outs = np.zeros((m,5,max_length))
    print('Converting to one-hot...')
    
    strains = []
    for i in range(0,m):
        entry = entries[i]
        
        strains.append(entry[1])
        one_hot_in = one_hot_sequence(entry[2])
        sequence_length = one_hot_in.shape[1]
        one_hot_ins[i,:,0:sequence_length] = one_hot_in
        
        one_hot_out = one_hot_sequence(entry[3])
        sequence_length = one_hot_out.shape[1]
        one_hot_outs[i,:,0:sequence_length] = one_hot_out
        
            
    print('Done')
    return one_hot_ins, one_hot_outs, strains


def load(datapath):
    print('Loading Data...')
    file = open(datapath, 'r')

    np.random.seed(0)
    train_entries = []
    test_entries = []
    val_entries = []
    max_length_in = 0
    max_length_out = 0

    for ln in file:
        toks = ln.split('\t')
        rand_num = np.random.random()
        if(toks[2] != toks[3]):
            max_length_in = max(max_length_in,len(toks[2]))
            max_length_out = max(max_length_out,len(toks[3]))
            
            if(rand_num < 0.95):
                train_entries.append(toks)
            elif(rand_num < 0.975):
                test_entries.append(toks)
            else:
                val_entries.append(toks)
            
        if toks[2] == toks[3]:
            if(rand_num > 0.975):
                val_entries.append(toks)
            elif(rand_num > 0.95):
                test_entries.append(toks)
            elif(rand_num > 0.82):
                train_entries.append(toks)

    file.close()

    num_entries = len(train_entries) + len(test_entries) + len(val_entries)
    
    # ANALYZE DATA
    print(str(num_entries) + ' sequences were uploaded')

    print('\nMaximum sequence length in is ' + str(max_length_in))
    print('Maximum sequence length out is ' + str(max_length_out) + '\n')
    
    # CONVERT DATA OR SUBSET OF DATA TO ONE-HOT
    [X_train, Y_train, target_strains_train] = one_hot_list(train_entries, max(max_length_in,max_length_out))
    [X_test, Y_test, target_strains_test] = one_hot_list(test_entries, max(max_length_in,max_length_out))
    [X_val, Y_val, target_strains_val] = one_hot_list(val_entries, max(max_length_in,max_length_out))
    
    return X_train, Y_train, target_strains_train, X_test, Y_test, target_strains_test, X_val,Y_val, target_strains_val

