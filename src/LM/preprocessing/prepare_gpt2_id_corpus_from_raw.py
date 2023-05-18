#from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
import torch
import random
import sys

import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)


input_dir = "./data/"
input_file = input_dir + "openwebtext_2017_18_small"
output_dir = ".//data/processed/openwebtext17-18_gpt2/tensors_all/"
output_train_file = output_dir + "train.pt"
output_val_file = output_dir + "val_org.pt"
output_test_file = output_dir + "test_org.pt"

training_ratio = 0.96
val_ratio = 0.02

max_line_num = 100000000000000
#max_line_num = 100000
#max_line_num = 10000000
#max_line_num = 20000000
#max_line_num = 2000000

#max_sent_len = 256

output_arr = []

#tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

i=0
with open(input_file, encoding='latin-1') as f_in:
    for line in f_in:
        raw_text = line
        i+=1
        indexed_tokens = tokenizer.encode(raw_text, add_prefix_space=True)
        output_arr.append(indexed_tokens)
        if i % 100000 == 0:
            print(i)
            sys.stdout.flush()
        if i > max_line_num:
            break

#idx_shuffled = list(range(len(output_arr)))
#random.shuffle(idx_shuffled)
training_size = int(len(output_arr)*training_ratio)
val_size = int(len(output_arr)*val_ratio)

def save_to_tensor(output_arr, output_file_name):
    data_size = len(output_arr)
    len_sum = 0
    for sent in output_arr:
        sent_len = len(sent)
        len_sum += sent_len
    #output_tensor = torch.zeros((len_sum),dtype = torch.uint16)
    output_tensor = torch.zeros((len_sum),dtype = torch.int32)

    current_start = 0
    for i in range(data_size):
        sent = output_arr[i]
        #output_tensor[current_start:current_start+len(sent)] = torch.tensor(sent,dtype = torch.uint16)
        output_tensor[current_start:current_start+len(sent)] = torch.tensor(sent,dtype = torch.int32)
        current_start += len(sent)

    torch.save(output_tensor, output_file_name)

save_to_tensor(output_arr[:training_size], output_train_file)
save_to_tensor(output_arr[training_size:training_size+val_size], output_val_file)
save_to_tensor(output_arr[training_size+val_size:], output_test_file)
