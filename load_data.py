import numpy as np

def one_hot_sequence(sequence):
    one_hot = np.zeros((1,4,len(sequence),1))

    for i in range(len(sequence)):
        if sequence[i] == 'A':
            one_hot[0,0,i,0] = 1
        elif sequence[i] == 'C':
            one_hot[0,1,i,0] = 1
        elif sequence[i] == 'G':
            one_hot[0,2,i,0] = 1
        elif sequence[i] == 'T':
            one_hot[0,3,i,0] = 1

    return one_hot

def one_hot_list(entries,max_length):
    m = len(entries)
    one_hot_ins = np.zeros((m,4,max_length,1))
    one_hot_outs = np.zeros((m,4,max_length,1))
    print('Converting to one-hot...')
    
    for i in range(0,m):
        entry = entries[i]
        
        one_hot_in = one_hot_sequence(entry[2])
        sequence_length = one_hot_in.shape[2]
        one_hot_ins[i,:,0:sequence_length] = one_hot_in
        
        one_hot_out = one_hot_sequence(entry[3])
        sequence_length = one_hot_out.shape[2]
        one_hot_outs[i,:,0:sequence_length] = one_hot_out
        
            
    print('Done')
    return one_hot_ins, one_hot_outs

def load(datapath):
    print('Loading data...')
    file = open(datapath, 'r')

    np.random.seed(0)
    train_entries = []
    test_entries = []
    max_length_in = 0
    max_length_out = 0

    for ln in file:
        toks = ln.split('\t')
        if(toks[2] != toks[3]):
            max_length_in = max(max_length_in,len(toks[2]))
            max_length_out = max(max_length_out,len(toks[3]))
            
            rand_num = np.random.random()
            if(rand_num < 0.95):
                train_entries.append(toks)
            else:
                test_entries.append(toks)
            
        # if toks[2] == toks[3]:
        #     same_entries.append(toks)
        # else:
        #     diff_entries.append(toks)

    file.close()
    # all_entries = same_entries + diff_entries
    # num_entries = len(all_entries)
    num_entries = len(train_entries) + len(test_entries)
    
    # ANALYZE DATA
    print(str(num_entries) + ' sequences were uploaded')

    print('\nMaximum sequence length in is ' + str(max_length_in))
    print('Maximum sequence length out is ' + str(max_length_out) + '\n')
    
    # CONVERT DATA OR SUBSET OF DATA TO ONE-HOT
    [X_train, Y_train] = one_hot_list(train_entries, max(max_length_in,max_length_out))
    [X_test, Y_test] = one_hot_list(test_entries, max(max_length_in,max_length_out))
    
    return X_train, Y_train, X_test, Y_test

