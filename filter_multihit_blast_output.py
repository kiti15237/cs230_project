#Read line, set that the current read_id
#Keep reading lines until you hit a line that is not that read id
#Deal with the set, selecting only the one with the max evalue
#Repeat
import pandas as pd
import numpy as np
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type = str, help = "Datapath for input data, format blast output tab delimited qid  ref_id  qseq  refseq  eval")
parser.add_argument('--out_file', type = str, help = "output file path")
parser.add_argument('--max_hits', type = int, help = "how many best hits to allow per read")
args = parser.parse_args()

#input_datapath = "../data/extreme_dataset/blast_tab_e-10_20hit_complete.out"
#output_datapath = "../data/extreme_dataset/blast_tab_e-10_20hit_complete_pruned.out"
input_datapath = args.in_file
output_datapath = args.out_file
max_hits = args.max_hits


def make_entry(ln):
    toks = ln.split('\t')
    read_id = toks[0]
    hit_strain = toks[1]
    query_seq = toks[2]
    hit_seq = toks[3]
    evalue = toks[4].strip("\n")
    return(pd.DataFrame({"evalue":evalue,  "hit_seq":hit_seq ,"query_seq":query_seq, 
                         "hit_strain": hit_strain, "read_id": read_id
                               }, index = [0]))
def save_max_values(working_set, final_set, entry):
    if working_set.shape[0] > 1:
        #Select highest evalue from working_set to add hit to final_set
        evalues = [float(x) for x in working_set['evalue']]
        max_val_ind = np.flatnonzero(evalues == np.min(evalues))
        selection = working_set.iloc[max_val_ind, :]
	num_hits = np.minimum(selection.shape[0], max_hits)
	rand_idx = np.random.randint(selection.shape[0], size=num_hits)
	selection = selection.iloc[rand_idx]
    else:
        selection = working_set
    final_set = pd.concat([final_set, selection])
    working_set = pd.DataFrame(entry) #start working set anew with the entry that triggered the change
    current_read_id = entry.iloc[0]['read_id']
    return final_set, working_set, current_read_id

def load_data(file_path):
    datapath = file_path
    #datapath = "../data/balanced_dataset/blast_tab_e-10_20hit_complete.out"
    #datapath = "../data/balanced_dataset/test_short2.out"
    print(datapath)
    file = open(datapath, 'r')

    prev_line = [None]*5
    read_hit_dict = {}
    streak = 0
    current_read_id = ""
    entry = make_entry(file.readline())
    current_read_id = entry.iloc[0]['read_id']

    working_set = entry
    final_set = pd.DataFrame()
    i = 0
    for ln in file:  
        i = i + 1
        if i % 10000 == 0:
            print(i)
        entry = make_entry(ln)
        read_id = entry.iloc[0]['read_id']
        if read_id == current_read_id:
            entry.index = [max(working_set.index) + 1]
            working_set = pd.concat([working_set, entry], axis = 0)
        else:
            #We have a new read_id
            final_set, working_set, current_read_id = save_max_values(working_set, final_set, entry)
            
    final_set, working_set, current_read_id = save_max_values(working_set, final_set, entry) #Save the last set
    return(final_set)

for file_path in os.listdir(input_datapath) :
	final_set = load_data(input_datapath + "/" +  file_path)
	final_set = final_set[["read_id", "hit_strain", "query_seq", "hit_seq", "evalue"]]
	print(final_set.shape)
	final_set.to_csv(output_datapath + "/" + file_path +"_onehit.out", sep = "\t", index = False, header = False)
