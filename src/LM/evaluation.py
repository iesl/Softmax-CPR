from gpt2_model.tokenization_gpt2 import GPT2Tokenizer
from utils_testing import MatrixReconstruction as MR
from sklearn.decomposition import TruncatedSVD
import torch
import numpy as np
import sys
import time

def only_test_time_loss(dataloader, GPT2_LM, parallel_GPT2_LM, args):
    GPT2_LM.eval()
    total_loss = 0.
    data_size = 0
    
    num_gpu = torch.cuda.device_count()
    spend_time_arr = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % 100 == 0:
                print(i_batch)
                sys.stdout.flush()
            if i_batch >= args.eval_batch_num:
                break
            feature = sample_batched
            batch_size = feature.size(0)
            t = time.time()
            outputs = parallel_GPT2_LM(feature, labels=feature)
            spend_time = time.time() - t
            if batch_size == args.batch_size:
                spend_time_arr.append(spend_time)
            loss = outputs[0]
            if num_gpu > 1:
                loss = loss.mean()
            total_loss += loss.item() * batch_size
            data_size += batch_size
    #data_size = len(dataloader.dataset)
    return total_loss / data_size, spend_time_arr
            

def simple_eval(dataloader, GPT2_LM, parallel_GPT2_LM, n_facet_effective, args):
    GPT2_LM.eval()
    num_gpu = torch.cuda.device_count()

    total_loss = 0.
    total_stage1_loss = 0.
    total_stage2_loss = 0.
    total_div = 0.
    data_size = 0

    total_weights_arr = np.zeros(n_facet_effective)
    total_count_best_arr = np.zeros(n_facet_effective)
    num_vis_bin = GPT2_LM.num_vis_bin
    num_vis_bin_loss = GPT2_LM.num_vis_bin_loss
    total_emb_div_arr_vis = np.zeros(num_vis_bin)
    total_loss_seq_vis = np.zeros(num_vis_bin_loss)
    total_stage1_loss_seq_vis = np.zeros(num_vis_bin_loss)
    total_stage2_loss_seq_vis = np.zeros(num_vis_bin_loss)
    total_MRR_seq_vis = np.zeros(num_vis_bin_loss)
    total_MRR = 0.

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % 100 == 0 and not args.template_experiment:
                print(i_batch)
                sys.stdout.flush()
            if i_batch >= args.eval_batch_num:
                break
            if args.template_experiment:
                feature, sent_len, target_position = sample_batched
                labels = -100 * torch.ones_like(feature)
                #labels[target_position:sent_len] = feature[target_position:sent_len]
                for j in range(feature.size(0)): #should be able to use gather and scatter if batch size is large
                    labels[j,target_position[j]:sent_len[j]] = feature[j,target_position[j]:sent_len[j]]
                    #labels[j,target_position[j]] = feature[j,target_position[j]]
            else:
                feature = sample_batched
                labels = feature
            #feature = sample_batched
            batch_size = feature.size(0)
            if args.n_facet == 0:
                outputs = parallel_GPT2_LM(feature, labels=labels)
            else:
                exclude_neg_labels_for_MRR = False
                if args.template_experiment:
                    exclude_neg_labels_for_MRR = True
                    
                if args.n_facet_stage2>0:
                    outputs, emb_div, count_best_arr, weight, emb_div_arr_vis, stage1_loss_seq_vis, stage2_loss_seq_vis, MRR_seq_vis, stage1_loss_raw, stage2_loss_raw, MRR_raw, div_raw = parallel_GPT2_LM(feature, labels=labels, vis_simple=True, exclude_neg_labels_for_MRR = exclude_neg_labels_for_MRR)
                    stage1_loss_seq_vis = stage1_loss_seq_vis.mean(dim=0)
                    stage2_loss_seq_vis = stage2_loss_seq_vis.mean(dim=0)
                    total_stage1_loss_seq_vis += stage1_loss_seq_vis.cpu().numpy() * batch_size
                    total_stage2_loss_seq_vis += stage2_loss_seq_vis.cpu().numpy() * batch_size
                else:
                    outputs, emb_div, count_best_arr, weight, emb_div_arr_vis, loss_seq_vis, MRR_seq_vis, loss_raw, MRR_raw, div_raw = parallel_GPT2_LM(feature, labels=labels, vis_simple=True, exclude_neg_labels_for_MRR = exclude_neg_labels_for_MRR)
                    loss_seq_vis = loss_seq_vis.mean(dim=0)
                    total_loss_seq_vis += loss_seq_vis.cpu().numpy() * batch_size
                        
                emb_div = emb_div.mean()
                weights_arr = weight.mean(dim=1).mean(dim=0)
                count_best_arr = count_best_arr.mean(dim=0)
                emb_div_arr_vis = emb_div_arr_vis.mean(dim=0)
                MRR_seq_vis = MRR_seq_vis.mean(dim=0)
                total_div += emb_div.item() * batch_size
                total_count_best_arr += count_best_arr.cpu().numpy() * batch_size
                total_emb_div_arr_vis += emb_div_arr_vis.cpu().numpy() * batch_size
                total_MRR_seq_vis += MRR_seq_vis.cpu().numpy() * batch_size
                total_MRR += MRR_raw.mean().item() * batch_size
                total_weights_arr += weights_arr.cpu().detach().numpy() * batch_size
            loss = outputs[0]
            if args.n_facet_stage2>0:
                stage1_loss = outputs[-2]
                stage2_loss = outputs[-1]
            if num_gpu > 1:
                loss = loss.mean()
                if args.n_facet_stage2>0:
                    stage1_loss = stage1_loss.mean()
                    stage2_loss = stage2_loss.mean()
            total_loss += loss.item() * batch_size
            if args.n_facet_stage2>0:
                total_stage1_loss += stage1_loss.item() * batch_size
                total_stage2_loss += stage2_loss.item() * batch_size
            data_size += batch_size
    #data_size = min(args.eval_batch_num*batch_size,len(dataloader.dataset))
    if args.n_facet_stage2>0:
        return total_loss / data_size, total_stage1_loss / data_size, total_stage2_loss / data_size, total_div / data_size, total_count_best_arr / data_size, total_weights_arr / data_size, total_emb_div_arr_vis / data_size, total_stage1_loss_seq_vis / data_size, total_stage2_loss_seq_vis / data_size, total_MRR_seq_vis / data_size, total_MRR / data_size
    else:
        return total_loss / data_size, total_div / data_size, total_count_best_arr / data_size, total_weights_arr / data_size, total_emb_div_arr_vis / data_size, total_loss_seq_vis / data_size, total_MRR_seq_vis / data_size, total_MRR / data_size
    

def comprehensive_evaluate(dataloader, GPT2_LM, parallel_GPT2_LM, n_facet_effective, args, device):
    # Turn on evaluation mode which disables dropout.
    GPT2_LM.eval()
    num_gpu = torch.cuda.device_count()
    total_loss = 0.
    total_div = 0.
    data_size = 0
        
    #total_count_arr = np.zeros(n_facet_effective)
    total_weights_arr = np.zeros(n_facet_effective)
    total_count_best_arr = np.zeros(n_facet_effective)
    num_vis_bin = GPT2_LM.num_vis_bin
    num_vis_bin_loss = GPT2_LM.num_vis_bin_loss
    total_emb_div_arr_vis = np.zeros(num_vis_bin)
    total_loss_seq_vis = np.zeros(num_vis_bin_loss)
    total_MRR_seq_vis = np.zeros(num_vis_bin_loss)
    reconstruction_err_arr = []
    reconstruction_err_norm_arr = []
    loss_raw_arr = []
    input_arr = []
    top_k_facet_idx_arr = []
    top_k_facet_prob_arr = []
    top_k_idx_arr = []
    top_k_prob_arr = []
    top_k_div_arr = []
    top_k_mag_arr = []
    top_k_div_norm_arr = []
    pred_div_arr = []
    target_norm_arr = []
    top_entropy_arr = []
    eig_vals_min_arr = []
    eig_vals_prod_arr = []
    eig_vals_min_norm_arr = []
    eig_vals_prod_norm_arr = []
    div_weighted_sq_arr = []
    weighted_sq_arr = []
    collapse_diff_arr = []
    collapse_diff_inv_arr = []
    collapse_diff_val_arr = []
    facet_norm_arr = []
    top_four_dist_arr = []
    top_val_single_arr = []
    top_idx_single_arr = []
    LID_avg_arr = []
    LID_avg_norm_arr = []

    model_name = 'gpt2'
    tokenizer_GPT2 = GPT2Tokenizer.from_pretrained(model_name)
    period_idx = tokenizer_GPT2.encode('.')[0] #suppose to be 13
    GPT2_LM.period_idx = period_idx
    loss_period_sum = torch.zeros(2*args.bptt, device=device, dtype=torch.float)
    MRR_period_sum = torch.zeros(2*args.bptt, device=device, dtype=torch.float)
    loss_period_count = torch.zeros(2*args.bptt, device=device, dtype=torch.long)

    all_pair_four = [ ([1,2],[3,4]), ([1,3],[2,4]), ([1,4],[2,3])]
    with torch.no_grad():
        output_word_emb_norm = GPT2_LM.lm_head.weight.data
        output_word_emb_norm = output_word_emb_norm / (1e-15 + output_word_emb_norm.norm(dim=-1,keepdim=True) )
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % 100 == 0:
                print(i_batch)
                sys.stdout.flush()
            if i_batch >= args.eval_batch_num:
                break
            feature = sample_batched
            batch_size = feature.size(0)
            if args.n_facet == 0:
                outputs = parallel_GPT2_LM(feature, labels=feature)
            else:
                #outputs, emb_div, count_arr, count_best_arr, emb_div_arr_vis, loss_seq_vis, MRR_seq_vis, loss_raw, MRR_raw, dist_to_period = parallel_GPT2_LM(feature, labels=feature)
                outputs, emb_div, count_best_arr, weight, emb_div_arr_vis, loss_seq_vis, MRR_seq_vis, loss_raw, MRR_raw, div_raw, dist_to_period, div_weighted_sq_raw, collapse_difference, collapse_difference_inv, collapse_difference_val, facet_norm, top_val_single, top_idx_single = parallel_GPT2_LM(feature, labels=feature, vis_seq_loss = True, eval_recon_top_k = args.eval_recon_top_k)
                if args.eval_recon_top_k > 1:

                    assert len(args.reconstruction_file_name) > 0
                    top_k = args.eval_recon_top_k
                    
                    pred_prob = outputs[1][:,:-1,:] #bsz, seq_len, vocab
                    labels = feature[:,1:]
                    top_val, top_idx = torch.topk(pred_prob, top_k, dim=2)
                    top_norm_prob = top_val / top_val.sum(dim=-1, keepdim=True)
                    top_entropy = -torch.sum(top_norm_prob * torch.log(top_norm_prob), dim = -1)
                    all_top_word_embs = GPT2_LM.lm_head.weight.data[top_idx,:] #bsz, seq_len, top_k, emb_size
                    all_top_word_embs_norm = output_word_emb_norm[top_idx,:] #bsz, seq_len, top_k, emb_size
                    bsz, seq_len, top_k, emb_size = all_top_word_embs_norm.size()
                    dist_ratio_arr = torch.zeros( (bsz, seq_len, len(all_pair_four)), dtype=float, device=output_word_emb_norm.device)
                    for j, (idx_pair_1, idx_pair_2) in enumerate(all_pair_four):
                        mean_dist = ( all_top_word_embs_norm[:,:,idx_pair_1, :].mean(dim=2) - all_top_word_embs_norm[:,:,idx_pair_2, :].mean(dim=2) ).norm(dim=-1)
                        pair_1_dist = ( all_top_word_embs_norm[:,:,idx_pair_1[0], :] - all_top_word_embs_norm[:,:,idx_pair_1[1], :] ).norm(dim=-1)
                        pair_2_dist = ( all_top_word_embs_norm[:,:,idx_pair_2[0], :] - all_top_word_embs_norm[:,:,idx_pair_2[1], :] ).norm(dim=-1)
                        dist_ratio_arr[:,:,j] = mean_dist / torch.maximum(pair_1_dist,pair_2_dist)
                    dist_ratio_arr = dist_ratio_arr.min(dim=-1)[0]
                    #emb_mean = all_top_word_embs_norm.mean(dim=2, keepdim=True)
                    #top_k_div_norm = torch.mean( (all_top_word_embs_norm - emb_mean).norm(dim=-1), dim=2 )
                    target = all_top_word_embs[:,:,0,:]    
                    target_norm = target.norm(dim=-1)
                    top_k_mag = all_top_word_embs.norm(dim=-1).mean(dim=-1)
                    bsz, seq_len, top_k, emb_size = all_top_word_embs.size()
                    #all_top_word_embs[:,:,:top_k,:top_k] += 1e-15 * torch.eye(top_k, dtype=float,device=all_top_word_embs.device).expand(bsz, seq_len, top_k, top_k) #prevent singular
                    try:
                        u, eig_vals, vh = torch.linalg.svd(all_top_word_embs, compute_uv=False)
                    except RuntimeError:
                        eig_vals = torch.ones((bsz, seq_len, top_k), dtype=float,device=all_top_word_embs.device)
                        eig_vals[:,:,0] = -1
                    #u, eig_vals, vh = torch.svd(all_top_word_embs, compute_uv=False)
                    eig_vals_min = eig_vals.min(dim=-1)[0]
                    eig_vals_prod = eig_vals.prod(dim=-1)

                    #all_top_word_embs_norm[:,:,:top_k,:top_k] += 1e-15 * torch.eye(top_k, dtype=float,device=all_top_word_embs_norm.device).expand(bsz, seq_len, top_k, top_k) #prevent singular
                    try:
                        u, eig_vals, vh = torch.linalg.svd(all_top_word_embs_norm, compute_uv=False)
                    except RuntimeError:
                        eig_vals = torch.ones((bsz, seq_len, top_k), dtype=float,device=all_top_word_embs.device)
                        eig_vals[:,:,0] = -1
                    eig_vals_min_norm = eig_vals.min(dim=-1)[0]
                    eig_vals_prod_norm = eig_vals.prod(dim=-1)
                    #eig_vals = np.linalg.svd(all_top_word_embs.cpu().detach().numpy(),compute_uv=False) #too slow
                    #svd = TruncatedSVD(n_components=top_k)
                    #svd.fit(all_top_word_embs.cpu().detach().numpy())
                    #eig_vals = svd.singular_values_

                    #eig_vals_min = np.amin(eig_vals, axis=-1)
                    #eig_vals_prod = np.prod(eig_vals,axis=-1)

                    def compute_div_recon(all_top_word_embs):
                        emb_mean = all_top_word_embs.mean(dim=2, keepdim=True)
                        emb_dist = (all_top_word_embs - emb_mean).norm(dim=-1)
                        top_k_div = torch.mean( emb_dist, dim=2 )
                        LID_avg = 1 / torch.mean( torch.log( emb_dist / emb_dist.max(dim=2,keepdim= True)[0] ) ,dim = 2 )

                        target = all_top_word_embs[:,:,0,:]
                        all_top_word_embs = all_top_word_embs[:,:,1:,:]
                        #label_expanded = labels.unsqueeze(-1).unsqueeze(-1).expand(bsz, seq_len, 1, emb_size) #label is not local in top k
                        #target = torch.gather(all_top_word_embs, dim=2, index=label_expanded ) # bsz, seq_len, 1, emb_size
                        #all_top_word_embs.scatter_(dim=2, index=label_expanded, src=0)
                        target = target.view(bsz*seq_len, 1, emb_size)
                        all_top_word_embs = all_top_word_embs.view(bsz*seq_len, top_k-1 , emb_size) #.transpose(1,2)
                        with torch.enable_grad():
                            mr = MR(batch_size*seq_len, 1, top_k-1, device=device)
                            lr = 0.05
                            max_iter = 200
                            opt = torch.optim.RMSprop(mr.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
                            loss_func = torch.nn.MSELoss(reduction='sum')
                            for i in range(max_iter):
                                opt.zero_grad()
                                pred = mr(all_top_word_embs)
                                loss = loss_func(pred, target) / 2
                                loss.backward()
                                opt.step()
                        target_pred = mr.coeff.matmul(all_top_word_embs)
                        loss_func = torch.nn.MSELoss(reduction='none')
                        reconstruction_err = loss_func(target_pred.view(bsz*seq_len, emb_size), target.view(bsz*seq_len, emb_size)).sum(dim=-1).view(bsz,seq_len) / 2 # bsz, seq_len
                        return top_k_div, reconstruction_err, LID_avg
                    top_k_div, reconstruction_err, LID_avg = compute_div_recon(all_top_word_embs)
                    top_k_div_norm, reconstruction_err_norm, LID_avg_norm = compute_div_recon(all_top_word_embs_norm)
                    reconstruction_err_list = reconstruction_err.view(-1).cpu().detach().tolist()
                    reconstruction_err_norm_list = reconstruction_err_norm.view(-1).cpu().detach().tolist()
                    loss_raw_list = loss_raw.view(-1).cpu().detach().tolist()
                    input_list = labels.reshape(-1).cpu().detach().tolist()
                    top_k_list = top_idx.view(bsz * seq_len, top_k).cpu().detach().tolist()
                    top_k_prob_list = top_val.view(bsz * seq_len, top_k).cpu().detach().tolist()
                    top_k_div_list = top_k_div.view(-1).cpu().detach().tolist()
                    top_k_div_norm_list = top_k_div_norm.view(-1).cpu().detach().tolist()
                    LID_avg_list = LID_avg.view(-1).cpu().detach().tolist()
                    LID_avg_norm_list = LID_avg_norm.view(-1).cpu().detach().tolist()
                    pred_div_list = div_raw.view(-1).cpu().detach().tolist()
                    target_norm_list = target_norm.view(-1).cpu().detach().tolist()
                    top_k_mag_list = top_k_mag.view(-1).cpu().detach().tolist()
                    top_entropy_list = top_entropy.view(-1).cpu().detach().tolist()
                    eig_vals_min_list = eig_vals_min.view(-1).cpu().detach().tolist()
                    eig_vals_prod_list = eig_vals_prod.view(-1).cpu().detach().tolist()
                    eig_vals_min_norm_list = eig_vals_min_norm.view(-1).cpu().detach().tolist()
                    eig_vals_prod_norm_list = eig_vals_prod_norm.view(-1).cpu().detach().tolist()
                    dist_ratio_list = dist_ratio_arr.view(-1).cpu().detach().tolist()
                    #eig_vals_min_list = eig_vals_min.reshape(-1).tolist()
                    #eig_vals_prod_list = eig_vals_prod.reshape(-1).tolist()
                    #reconstruction_err_arr += zip(reconstruction_err_list, loss_raw_list)
                    top_four_dist_arr += dist_ratio_list
                    reconstruction_err_arr += reconstruction_err_list
                    reconstruction_err_norm_arr += reconstruction_err_norm_list
                    loss_raw_arr += loss_raw_list
                    input_arr += input_list
                    top_k_idx_arr += top_k_list
                    top_k_prob_arr += top_k_prob_list
                    top_k_div_arr += top_k_div_list
                    top_k_div_norm_arr += top_k_div_norm_list
                    LID_avg_arr += LID_avg_list
                    LID_avg_norm_arr += LID_avg_norm_list
                    pred_div_arr += pred_div_list
                    target_norm_arr += target_norm_list
                    top_k_mag_arr += top_k_mag_list
                    top_entropy_arr += top_entropy_list
                    eig_vals_min_arr += eig_vals_min_list
                    eig_vals_prod_arr += eig_vals_prod_list
                    eig_vals_min_norm_arr += eig_vals_min_norm_list
                    eig_vals_prod_norm_arr += eig_vals_prod_norm_list
                    if top_val_single is None:
                        top_val_single_arr += [[0]*top_k] * len(input_list)
                        top_idx_single_arr += [[0]*top_k] * len(input_list)
                    else:
                        top_val_single_arr += top_val_single[:,:-1,:].reshape(bsz * seq_len, top_k).cpu().detach().tolist()
                        top_idx_single_arr += top_idx_single[:,:-1,:].reshape(bsz * seq_len, top_k).cpu().detach().tolist()
                        
                        stacked_facet_lm_logits = outputs[2]
                        top_value_facet, top_idx_facet = torch.topk( stacked_facet_lm_logits[:, :, :,:].softmax(dim=-1), k=top_k, dim=-1 )
                        top_k_facet_prob_arr += top_value_facet[:,:,:-1,:].reshape(n_facet_effective, bsz * seq_len, top_k).permute(1,0,2).cpu().detach().tolist()
                        top_k_facet_idx_arr += top_idx_facet[:,:,:-1,:].reshape(n_facet_effective, bsz * seq_len, top_k).permute(1,0,2).cpu().detach().tolist()
                        
                    if collapse_difference is None:
                        collapse_diff_arr += [0] * len(input_list)
                        collapse_diff_inv_arr += [0] * len(input_list)
                        collapse_diff_val_arr += [0] * len(input_list)
                    else:
                        collapse_diff_arr += collapse_difference[:,:-1].reshape(-1).cpu().detach().tolist()
                        collapse_diff_inv_arr += collapse_difference_inv[:,:-1].reshape(-1).cpu().detach().tolist()
                        collapse_diff_val_arr += collapse_difference_val[:,:-1].reshape(-1).cpu().detach().tolist()

                    if facet_norm is None:
                        facet_norm_arr += [0] * len(input_list)
                    else:
                        facet_norm_arr += facet_norm[:,:-1].reshape(-1).cpu().detach().tolist()
                        
                    if n_facet_effective == args.n_facet:
                        div_weighted_sq_list = div_weighted_sq_raw[:,:-1].reshape(-1).cpu().detach().tolist()
                        div_weighted_sq_arr += div_weighted_sq_list
                    else:
                        div_weighted_sq_arr += [0]*len(input_list)
                    assert len(collapse_diff_arr) == len(input_arr) == len(div_weighted_sq_arr) == len(facet_norm_arr) 
                    #bsz, seq_len
                    #torch.gather( GPT2_LM.lm_head.weight.data, dim=-1, index=top_idx, )
                dist_to_period_pos = dist_to_period
                dist_to_period_pos[dist_to_period_pos<0] = 2*args.bptt + dist_to_period_pos[dist_to_period_pos<0]
                #print(dist_to_period)
                loss_period_sum.scatter_add_(0, dist_to_period_pos, loss_raw)
                MRR_period_sum.scatter_add_(0, dist_to_period_pos, MRR_raw)
                loss_period_count.scatter_add_(0, dist_to_period_pos, torch.ones_like(loss_raw,dtype=torch.long))
                #if num_gpu > 1:
                emb_div = emb_div.mean()
                #count_arr = count_arr.mean(dim=0)
                weights_arr = weight.mean(dim=1).mean(dim=0)
                count_best_arr = count_best_arr.mean(dim=0)
                emb_div_arr_vis = emb_div_arr_vis.mean(dim=0)
                loss_seq_vis = loss_seq_vis.mean(dim=0)
                MRR_seq_vis = MRR_seq_vis.mean(dim=0)
                total_div += emb_div.item() * batch_size
                #total_count_arr += count_arr.cpu().numpy() * batch_size
                total_weights_arr += weights_arr.cpu().detach().numpy() * batch_size
                total_count_best_arr += count_best_arr.cpu().numpy() * batch_size
                total_emb_div_arr_vis += emb_div_arr_vis.cpu().numpy() * batch_size
                total_loss_seq_vis += loss_seq_vis.cpu().numpy() * batch_size
                total_MRR_seq_vis += MRR_seq_vis.cpu().numpy() * batch_size
            loss = outputs[0]
            if num_gpu > 1:
                loss = loss.mean()
            total_loss += loss.item() * batch_size
            data_size += batch_size

    import gzip
    import pickle
    #data_size = min(args.eval_batch_num*batch_size,len(dataloader.dataset))
    if args.eval_recon_top_k > 1:
        #with gzip.GzipFile(args.reconstruction_file_name+'.pk.gz', 'w') as f_out:
        #    f_out.write(pickle.dumps( (input_arr, loss_raw_arr, reconstruction_err_arr, reconstruction_err_norm_arr, top_k_div_arr, top_k_div_norm_arr, pred_div_arr, div_weighted_sq_arr, target_norm_arr, top_k_mag_arr, top_entropy_arr, eig_vals_min_arr, eig_vals_prod_arr, eig_vals_min_norm_arr, eig_vals_prod_norm_arr, collapse_diff_arr, collapse_diff_inv_arr, collapse_diff_val_arr, facet_norm_arr, top_four_dist_arr, LID_avg_arr, LID_avg_norm_arr, top_k_idx_arr, top_k_prob_arr, top_idx_single_arr, top_val_single_arr) ))
        with open(args.reconstruction_file_name, 'w') as f_out:
            for i in range(len(loss_raw_arr)):
                f_out.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} ".format(input_arr[i], loss_raw_arr[i], reconstruction_err_arr[i], reconstruction_err_norm_arr[i], top_k_div_arr[i], top_k_div_norm_arr[i], pred_div_arr[i], div_weighted_sq_arr[i], target_norm_arr[i], top_k_mag_arr[i], top_entropy_arr[i], eig_vals_min_arr[i], eig_vals_prod_arr[i], eig_vals_min_norm_arr[i], eig_vals_prod_norm_arr[i], collapse_diff_arr[i], collapse_diff_inv_arr[i], collapse_diff_val_arr[i], facet_norm_arr[i], top_four_dist_arr[i], LID_avg_arr[i], LID_avg_norm_arr[i]))
                f_out.write(" ".join(str(x) for x in top_k_idx_arr[i]))
                f_out.write(" "+" ".join(str(x) for x in top_k_prob_arr[i]))
                f_out.write(" "+" ".join(str(x) for x in top_idx_single_arr[i]))
                f_out.write(" "+" ".join(str(x) for x in top_val_single_arr[i]))
                if len(top_k_facet_idx_arr) > 0:
                    for j in range(len(top_k_facet_idx_arr[i])):
                        f_out.write(" "+" ".join(str(x) for x in top_k_facet_idx_arr[i][j]))
                        f_out.write(" "+" ".join(str(x) for x in top_k_facet_prob_arr[i][j]))
                f_out.write("\n")
    #return total_loss / data_size, total_div / data_size, total_count_arr / data_size, total_count_best_arr / data_size, total_emb_div_arr_vis / data_size, total_loss_seq_vis / data_size, total_MRR_seq_vis / data_size, loss_period_sum.cpu().numpy(), loss_period_count.cpu().numpy(), MRR_period_sum.cpu().numpy()
    return total_loss / data_size, total_div / data_size, total_count_best_arr / data_size, total_weights_arr / data_size, total_emb_div_arr_vis / data_size, total_loss_seq_vis / data_size, total_MRR_seq_vis / data_size, loss_period_sum.cpu().numpy(), loss_period_count.cpu().numpy(), MRR_period_sum.cpu().numpy()
