import spacy
import torch
from spacy.lang.en import English
import sys
import time

from transformers import GPT2Tokenizer
from result_statistics import result_statistics
import torch.nn as nn

class MatrixReconstruction(nn.Module):
    def __init__(self, batch_size, nbow, ntopic, device):
        super(MatrixReconstruction, self).__init__()
        self.coeff = nn.Parameter(torch.randn(batch_size, nbow, ntopic, device=device, requires_grad=True))
        self.device = device
    
    #def compute_coeff_pos(self):
    #    self.coeff.data = self.coeff.clamp(0.0, 1.0)
    
    def forward(self, input):
        #print(input.size())
        #print(self.coeff.size())
        result = self.coeff.matmul(input)
        return result

def preprocessing_context(context, outf = None, shortest_context = 50, longest_context = 150):
    bad_context = False
    context = context.replace('â',"'").replace('â','-').replace('\n'," ")
    if len(context.split()) < shortest_context:
    #if len(context.split()) < 5:
        if outf is not None:
            outf.write("Skip due to short context\n")
        bad_context = True
    #if len(context.split()) > 50:
    if len(context.split()) > longest_context:
        if outf is not None:
            outf.write("Skip due to long context\n")
        bad_context = True
    try:
        context.encode('ascii', 'strict')
    except:
        if outf is not None:
            outf.write("Skip due to special token\n")
        bad_context = True
    return context, bad_context

def print_sampled_sent(generated_sent, outf, print_prefix):
    outf.write(print_prefix + ': ' + generated_sent + '\n')

def print_eval_text(feature, i_batch, outf, tokenizer_GPT2, inner_idx_tensor, gen_sent_tensor, gen_sent_tensor_org, result_stats, word_raw_list, word_raw_rest_list, readable_context, run_eval):
    #batch_size, num_head, top_k, n_basis = top_index.size()
    #num_sent_gen = gen_sent_tensor.size(2)
    batch_size, num_head, num_sent_gen, gen_sent_len = gen_sent_tensor.size()
    
    #not_ascii_multi = 0
    #not_ascii_org = 0
    for i_sent in range(batch_size):
        outf.write('batch number: ' + str(i_sent) + '\n')
        last_end = -1
        # for m in range(1):
        for m in range(num_head):
            outf.write('number of head: ' + str(m) + '\n')
            end = inner_idx_tensor[i_sent,m].item()
            if end == last_end:
                continue
            last_end = end
            
            #outf.write(tokenizer_GPT2.convert_tokens_to_string(feature_text[i_sent][:end])+'\n')
            context = tokenizer_GPT2.decode(feature[i_sent,:end])
            
            if readable_context:
                context, bad_context = preprocessing_context(context, outf)
                if bad_context:
                    continue

            outf.write(context+'\n')
            outf.write('\n')
            
            multi_sent_list = []
            org_sent_list = []
            for j in range(num_sent_gen):
                generated_sent = tokenizer_GPT2.decode( gen_sent_tensor[i_sent, m, j, :] )
                multi_sent_list.append(generated_sent)
                if gen_sent_tensor_org.size(0) > 0:
                    generated_sent_org = tokenizer_GPT2.decode( gen_sent_tensor_org[i_sent, m, j, :] )
                    org_sent_list.append(generated_sent_org)

            for j in range(num_sent_gen):
                generated_sent = multi_sent_list[j]
                print_sampled_sent(generated_sent, outf, 'multi-facet '+ str(j))
                if run_eval:
                    result_stats.update("Multi-facet", gen_sent_tensor[i_sent, m, j, :], feature[i_sent,:end], tokenizer_GPT2, word_raw_list[i_sent][m], word_raw_rest_list[i_sent][m], m)
            if run_eval:
                result_stats.update_self_BLEU("Multi-facet", m)
            if gen_sent_tensor_org.size(0) > 0:
                for j in range(num_sent_gen):
                    
                    generated_sent_org = org_sent_list[j]
                    print_sampled_sent(generated_sent_org, outf, 'single-facet '+ str(j))
                    if run_eval:
                        result_stats.update("Single-facet", gen_sent_tensor_org[i_sent, m, j, :], feature[i_sent,:end], tokenizer_GPT2, word_raw_list[i_sent][m], word_raw_rest_list[i_sent][m], m)
                if run_eval:
                    result_stats.update_self_BLEU("Single-facet", m)

            if run_eval:
                result_stats.renew_ngram(m)
    #outf.write('Number of not ascii code in multi: {}, in single: {}: {}\n'.format(not_ascii_conditional, not_ascii_org))


def top_k_logits(logits, k, filling_value=-1e10):
    #modified from https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/sample.py
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * filling_value, logits)

#def sample_seq_prob(model, context, gen_sent_len, device, top_k = 40, sample=True):
def sample_seq_prob(model, is_context, context, gen_sent_len, device, top_k = 10, sample=True):
#def sample_seq(model_condition, context, insert_loc, future_emb_chosen_arr, gen_sent_len, device, temperature=1, top_k = 5, sample=True):
    #modified from https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/sample.py
    prev = context
    batch_size = prev.size(0)
    output = torch.zeros((batch_size, 0), dtype=torch.long, device = device)
    past = None
    prev_input_ids = None
    prev_hidden_states = None #torch.zeros((batch_size, 0, 768), dtype=torch.long, device = device)
    #sample = False
    for i in range(gen_sent_len):
        #modify to pass past input ids and hidden_states
        if is_context:
            outputs = model(prev, past=past, all_input_ids = prev_input_ids, prev_hidden_states = prev_hidden_states)
        else:
            outputs = model(prev, past=past)
        #outputs[0] : prob, ..., transformer_outputs[1:]. transformer_outputs: last_hidden_state, past, hidden_state, ...
        probs = outputs[0][0]
        past = outputs[0][2]
        all_hidden_states = outputs[0][3] #tuple of 13 items, (bsz, seq_len, hid_dim)
        probs = probs[:, -1, :] 
        #print(probs.sum(dim=-1))
        probs = top_k_logits(probs, k=top_k, filling_value=0)
        #print(probs.sum(dim=-1))
        if i ==0:
            prev_hidden_states = all_hidden_states
            prev_input_ids = context
        else:
            prev_input_ids = torch.cat((prev_input_ids, prev), dim=1)
            tuple_temp =tuple()
            for layer, _ in enumerate(prev_hidden_states):                
                tuple_temp += (torch.cat((prev_hidden_states[layer], all_hidden_states[layer]), dim=1), )
            prev_hidden_states = tuple_temp
        #probs = F.softmax(logits, dim=-1)
        if sample:
            if torch.isnan(probs).sum() > 0:
                print(past)
                print(prev)
                print(insert_loc)
                print(future_emb_chosen_arr)
                print(logits)
                print(probs)
                sys.exit(0)
            prev = torch.multinomial(probs, num_samples=1)
        else:
            _, prev = torch.topk(probs, k=1, dim=-1)        
        
        # print(i, ": ",output.size(), len(all_hidden_states), all_hidden_states[-1].size()) #(seq_len, 1/2/3...) 13 (bsz, seq_len, 768)
        # for k in all_hidden_states:
        #     print(k.size())
        output = torch.cat((output, prev), dim=1)
    return output

def sample_seq(model, is_context, context, gen_sent_len, device, temperature=1, top_k = 40, sample=True):
#def sample_seq(model_condition, context, insert_loc, future_emb_chosen_arr, gen_sent_len, device, temperature=1, top_k = 5, sample=True):
    #modified from https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/sample.py
    prev = context
    batch_size = prev.size(0)
    output = torch.zeros((batch_size, 0), dtype=torch.long, device = device)
    past = None
    prev_hidden_states = None #torch.zeros((batch_size, 0, 768), dtype=torch.long, device = device)
    prev_input_ids = None
    #sample = False
    for i in range(gen_sent_len):
        # outputs = model(prev, past=past)
        #modify to pass past input ids and hidden_states
        # outputs = model(prev, past=past, all_input_ids = prev_input_ids, prev_hidden_states = prev_hidden_states)
        if is_context:
            outputs = model(prev, past=past, all_input_ids = prev_input_ids, prev_hidden_states = prev_hidden_states)
        else:
            outputs = model(prev, past=past)
        #outputs[0] : prob, ..., transformer_outputs[1:]. transformer_outputs: last_hidden_state, past, hidden_state, ...

        logits = outputs[0][0]
        past = outputs[0][2]
        logits = logits[:, -1, :] / temperature
        logits = top_k_logits(logits, k=top_k)
        all_hidden_states = outputs[0][3] #tuple of 13 items, (bsz, seq_len, hid_dim)
        if i ==0:
            prev_hidden_states = all_hidden_states
            prev_input_ids = context
        else:
            prev_input_ids = torch.cat((prev_input_ids, prev), dim=1)
            tuple_temmp = tuple()
            for layer, _ in enumerate(prev_hidden_states):                
                tuple_temmp += (torch.cat((prev_hidden_states[layer], all_hidden_states[layer]), dim=1), )
            prev_hidden_states = tuple_temmp

        probs = F.softmax(logits, dim=-1)
        if sample:
            if torch.isnan(probs).sum() > 0:
                print(past)
                print(prev)
                print(insert_loc)
                print(future_emb_chosen_arr)
                print(logits)
                print(probs)
                sys.exit(0)
            prev = torch.multinomial(probs, num_samples=1)        
        else:
            _, prev = torch.topk(probs, k=1, dim=-1)        
        output = torch.cat((output, prev), dim=1)
    return output


def get_word_list_spacy(inner_idx_tensor, feature_text, tokenizer_GPT2, nlp):
    def get_word_list_from_text(feature_text_i):
        feature_text_i_str = tokenizer_GPT2.convert_tokens_to_string(feature_text_i)
        tokens = nlp.tokenizer(feature_text_i_str)
        word_raw_list_i_j = []

        for tok in tokens:
            w = tok.text
            word_raw_list_i_j.append(w)
            
        return word_raw_list_i_j
    word_raw_list = []
    word_raw_rest_list = []
    batch_size, num_head = inner_idx_tensor.size()
    inner_idx_tensor_np = inner_idx_tensor.cpu().numpy()
    for b, feature_text_i in enumerate(feature_text):
        word_raw_list_i = []
        word_raw_rest_list_i = []
        for j in range(num_head):
            end_idx = inner_idx_tensor_np[b,j]
            word_raw_list_i_j = get_word_list_from_text(feature_text_i[:end_idx])
            #assert len(word_idx_list_i_j) > 0, print(feature_text_i[:end_idx])
            word_raw_list_i.append(word_raw_list_i_j)
            if end_idx == len(feature_text_i):
                word_raw_rest_list_i.append([])
            else:
                word_raw_rest_list_i_j = get_word_list_from_text(feature_text_i[end_idx:])
                word_raw_rest_list_i.append(word_raw_rest_list_i_j)
            #count = word_idx_d2_count.get(w_idx,0)
            #word_idx_d2_count[w_idx] += 1
        word_raw_list.append(word_raw_list_i)
        word_raw_rest_list.append(word_raw_rest_list_i)
    return word_raw_list, word_raw_rest_list


def visualize_interactive_LM(model_multi, is_context_multi, model_single, is_context_single, gpt2_model, device, num_sent_gen, gen_sent_len, dataloader, outf, max_batch_num, tokenizer_GPT2, bptt, readable_context = False, run_eval = True):
    top_k = 5
    nlp = English()

    #emb_sum = torch.sum(word_norm_emb,dim=1)
    #OOV_list = torch.nonzero(emb_sum == 0).squeeze().cpu().tolist()
    #print("OOV number = {}".format(len(OOV_list)))
    #print("OOV index examples {}".format(OOV_list[:10]))
    #OOV_set = set(OOV_list)

    with torch.no_grad():
        if run_eval:
            result_stats = result_statistics(gpt2_model)
            result_stats.add_model("Multi-facet")
            result_stats.add_model("Single-facet")
        else:
            result_stats = []
        for i_batch, sample_batched in enumerate(dataloader):
            # if i_batch == 0:
            #     continue
            print("batch"+str(i_batch))
            sys.stdout.flush()
            
            feature = sample_batched
            
            feature_text = [ [tokenizer_GPT2._convert_id_to_token(x) for x in feature[i,:].tolist()] for i in range(feature.size(0))]

            max_prompt_len = bptt - gen_sent_len
            min_prompt_len = 20
            interval = gen_sent_len
            
            batch_size = feature.size(0)
            start_idx_list = list(range(min_prompt_len,max_prompt_len,interval))
            num_head = len(start_idx_list)
            inner_idx_tensor = torch.tensor(start_idx_list, dtype=torch.long, device = device)
            inner_idx_tensor = inner_idx_tensor.expand(batch_size, num_head)
            batch_size, num_head = inner_idx_tensor.size()

            gen_sent_tensor = torch.empty( (batch_size, num_head, num_sent_gen, gen_sent_len), dtype=torch.long, device=device )
            gen_sent_tensor_org = torch.empty( (batch_size, num_head, num_sent_gen, gen_sent_len), dtype=torch.long, device=device )

            word_raw_list, word_raw_rest_list = get_word_list_spacy(inner_idx_tensor, feature_text, tokenizer_GPT2, nlp)
            # for i_sent in range(1):
            for i_sent in range(batch_size):
                print("sent"+str(i_sent))
                last_end = -1
                
                # for m in range(1):
                for m in range(num_head):
                    print("head"+str(m))

                    end = inner_idx_tensor[i_sent,m]
                    if end == last_end:
                        continue
                    last_end = end
                    
                    context = tokenizer_GPT2.convert_tokens_to_string(feature_text[i_sent][:end])
                    if readable_context:
                        context_proc, bad_context = preprocessing_context(context, outf)
                        if bad_context:
                            continue

                    end_int = end.item()
                    
                    start_int = 0
                    #if end_int > max_prompt_len:
                    #    start_int = end_int - max_prompt_len
                    
                    t = time.time()
                    feature_expanded = feature[i_sent,start_int:end].unsqueeze(0).expand(num_sent_gen,end_int - start_int).to(device = device)
                    if model_multi.output_probs:
                        output = sample_seq_prob(model_multi, is_context_multi, feature_expanded, gen_sent_len, device)
                    else:
                        output = sample_seq(model_multi, is_context_multi, feature_expanded, gen_sent_len, device)
                    multi_elapsed = time.time() - t
                    gen_sent_tensor[i_sent, m, :, :] = output
                    
                    t = time.time()
                    if model_single.output_probs:
                        output_org = sample_seq_prob(model_single, is_context_single, feature_expanded, gen_sent_len, device)
                    else:
                        output_org = sample_seq(model_single, is_context_single, feature_expanded, gen_sent_len, device)
                    org_elapsed = time.time() - t
                    gen_sent_tensor_org[i_sent, m, :, :] = output_org
                    
                    
                    if run_eval:
                        #result_stats.model_results["time_count"] += 1
                        for method_name, time_spent in [ ("Multi-facet",multi_elapsed), ("Single-facet", org_elapsed)]:
                            result_stats.model_results[method_name]["time_sum"] += time_spent
                            result_stats.model_results[method_name]["time_count"] += 1

                
            print_eval_text(feature, i_batch, outf, tokenizer_GPT2, inner_idx_tensor, gen_sent_tensor, gen_sent_tensor_org, result_stats, word_raw_list, word_raw_rest_list,  readable_context, run_eval)
            if i_batch + 1 >= max_batch_num:
                break
        if run_eval:
            result_stats.print()
            result_stats.generate_report(outf)