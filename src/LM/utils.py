import torch
import torch.utils.data
import numpy as np
import os, shutil
import random
import time
import datetime

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, sent_tensor, sent_len_tensor, target_position_tensor, device):
        self.sent_tensor = sent_tensor
        self.sent_len_tensor = sent_len_tensor
        self.target_position_tensor = target_position_tensor
        self.output_device = device

    def __len__(self):
        return int( self.sent_tensor.size(0) )
    
    def __getitem__(self, idx):
        feature = self.sent_tensor[idx,:].to(dtype = torch.long, device = self.output_device)
        sent_len = self.sent_len_tensor[idx]
        target_position = self.target_position_tensor[idx]
        return feature, sent_len, target_position

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, w_ind_gpt2_tensor, bptt, device):
        self.w_ind_gpt2 = w_ind_gpt2_tensor
        self.seq_len = bptt
        self.output_device = device

    def __len__(self):
        return int( self.w_ind_gpt2.size(0) /self.seq_len )
    
    def __getitem__(self, idx):
        feature = self.w_ind_gpt2[idx*self.seq_len:(idx+1)*self.seq_len].to(dtype = torch.long, device = self.output_device)
        return feature

def create_data_loader_split(f_in, bsz, bptt,  device, split_num, dataset_class):
    w_ind_gpt2_tensor  = torch.load(f_in, map_location='cpu')
    training_size = w_ind_gpt2_tensor.size(0)
    #idx_arr = np.random.permutation(training_size)
    split_size = int(training_size / split_num)
    dataset_arr = []
    for i in range(split_num):
        start = i * split_size
        if i == split_num - 1:
            end = training_size
        else:
            end = (i+1) * split_size
        dataset_arr.append(  dataset_class(w_ind_gpt2_tensor[start:end], bptt, device ) ) #assume that the dataset are randomly shuffled beforehand

    use_cuda = False
    if device.type == 'cuda':
        use_cuda = True
    #dataloader_arr = [torch.utils.data.DataLoader(dataset_arr[i], batch_size = bsz, shuffle = True, pin_memory=use_cuda, drop_last=False) for i in range(split_num)]
    dataloader_arr = [torch.utils.data.DataLoader(dataset_arr[i], batch_size = bsz, shuffle = True, pin_memory=not use_cuda, drop_last=True) for i in range(split_num)]
    return dataloader_arr

def create_data_loader(f_in, bsz, bptt, device, dataset_class, want_to_shuffle = True, is_evaluation = False):
    w_ind_gpt2_tensor = torch.load(f_in, map_location='cpu')
    #81816777
    #if is_evaluation:
    #    print("eval_data_size:", w_ind_gpt2_tensor.size(0))
    #    w_ind_gpt2_tensor = w_ind_gpt2_tensor[:100000]
    #    print("eval_data_size:", w_ind_gpt2_tensor.size(0))
    dataset = dataset_class(w_ind_gpt2_tensor, bptt, device)
    use_cuda = False
    if device.type == 'cuda':
        use_cuda = True
    #return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = want_to_shuffle, pin_memory=use_cuda, drop_last=False)
    return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = want_to_shuffle, pin_memory=not use_cuda, drop_last=True)

def create_sent_data_loader(f_in, bsz, device, dataset_class, want_to_shuffle = True):
    sent_tensor, sent_len_tensor, target_position_tensor = torch.load(f_in, map_location='cpu')
    dataset = dataset_class(sent_tensor, sent_len_tensor, target_position_tensor, device)
    use_cuda = False
    if device.type == 'cuda':
        use_cuda = True
    return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = want_to_shuffle, pin_memory=not use_cuda, drop_last=False)
    #return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = want_to_shuffle, pin_memory=not use_cuda, drop_last=True)

def load_sent_corpus(data_path, relation_type, testing_file_names_list, train_bsz, eval_bsz, device, skip_training = False, want_to_shuffle_val = False, load_testing = True):

    val_org_corpus_name = os.path.join(data_path, relation_type + '_one_overlap')
    #val_org_corpus_name = os.path.join(data_path, relation_type + '_no_overlap')

    dataset_class = SentenceDataset

    with open(val_org_corpus_name,'rb') as f_in:
        dataloader_val = create_sent_data_loader(f_in, eval_bsz, device, dataset_class, want_to_shuffle = want_to_shuffle_val)

    if load_testing:
        if len(testing_file_names_list) == 0 or len(testing_file_names_list[0])==0:
            test_org_corpus_name = os.path.join(data_path, relation_type + '_no_overlap')
            #test_org_corpus_name = os.path.join(data_path, relation_type + '_one_overlap')
            with open(test_org_corpus_name,'rb') as f_in:
                dataloader_test_arr = [create_sent_data_loader(f_in, eval_bsz, device, dataset_class, want_to_shuffle = want_to_shuffle_val)]
        else:
            dataloader_test_arr = []
            for test_org_corpus_name in testing_file_names_list:
                test_org_corpus_name = os.path.join(data_path, test_org_corpus_name + '_no_overlap')
                with open(test_org_corpus_name,'rb') as f_in:
                    dataloader_test_arr.append(create_sent_data_loader(f_in, eval_bsz, device, dataset_class, want_to_shuffle = want_to_shuffle_val))

    if skip_training:
        dataloader_train_arr = [0]
    else:
        train_corpus_name = os.path.join(data_path, relation_type + '_train')
        with open(train_corpus_name,'rb') as f_in:
            dataloader_train_arr = [create_sent_data_loader(f_in, train_bsz, device, dataset_class)]

    if load_testing:
        return dataloader_train_arr, dataloader_val, dataloader_test_arr
    else:
        return dataloader_train_arr, dataloader_val

def load_corpus(data_path, train_bsz, eval_bsz, bptt, device, tensor_folder = "tensors", split_num = 1, skip_training = False, want_to_shuffle_val = False, load_testing = False):
    train_corpus_name = data_path + "/" + tensor_folder + "/train.pt"
    # train_corpus_name = data_path + "/" + tensor_folder + "/val_org.pt" #for debugging
    val_org_corpus_name = data_path +"/" + tensor_folder + "/val_org.pt"
    #dictionary_input_name = data_path + "/" + tensor_folder + "/dict_idx_compact"

    dataset_class = SeqDataset

    with open(val_org_corpus_name,'rb') as f_in:
        dataloader_val = create_data_loader(f_in, eval_bsz, bptt, device, dataset_class, want_to_shuffle = want_to_shuffle_val, is_evaluation=True)

    if load_testing:
        test_org_corpus_name = data_path +"/" + tensor_folder + "/test_org.pt"
        with open(test_org_corpus_name,'rb') as f_in:
            dataloader_test = create_data_loader(f_in, eval_bsz, bptt, device, dataset_class, want_to_shuffle = want_to_shuffle_val)

    if skip_training:
        dataloader_train_arr = [0]
    else:
        with open(train_corpus_name,'rb') as f_in:
            dataloader_train_arr = create_data_loader_split(f_in, train_bsz, bptt, device, split_num, dataset_class)

    if load_testing:
        return dataloader_train_arr, dataloader_val, dataloader_test
    else:
        return dataloader_train_arr, dataloader_val

def seed_all_randomness(seed,use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not use_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
