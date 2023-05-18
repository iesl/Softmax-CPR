from transformers.modeling_outputs import BaseModelOutput
from torch.nn import CrossEntropyLoss
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.utils import ModelOutput
import torch.nn.functional as F
import torch.nn as nn
import copy
import math
from torch.nn import CrossEntropyLoss

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5PreTrainedModel, T5Stack
from generation import GenerationMixin_updated

class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states_all: Optional[Tuple[torch.FloatTensor]] = None #ADD Haw-Shiuan
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_outputs: Optional = None

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
        
    
def dynamic_partition_softmax_init(model, config, n_facet_all = 1, n_facet=1, n_facet_context =0, n_facet_reranker =0, n_facet_local_encoder = 0, n_facet_hidden=1, n_facet_window=0, n_facet_MLP=0, use_proj_bias=False, weight_mode = '',softmax_nonlinear='None', context_efficient_mode='assign_average_of_duplicates', reranker_efficient_mode='assign_only_last_of_duplicates', context_source=['encode', 'decode'], reranker_CAN_NUM=[20], candidates_from_previous_reranker=True, pointer_mode='' ):
        
        if n_facet_all == n_facet + n_facet_context*len(context_source) + n_facet_reranker*len(reranker_CAN_NUM) + n_facet_local_encoder:
            use_lm_head_context = False
        else:
            use_lm_head_context = True

        assert n_facet_context <= n_facet
        model.use_lm_head_context = use_lm_head_context
        if model.use_lm_head_context:
            #model.n_facet_emb = 1
            model.n_facet_emb = 2
        else:
            model.n_facet_emb = 0
        
        if len(pointer_mode) > 0:
            assert n_facet_local_encoder == 1

        if n_facet_local_encoder > 0 and len(pointer_mode) == 0:
            assert n_facet_local_encoder == n_facet_context
        assert n_facet + n_facet_context*len(context_source) + n_facet_reranker*len(reranker_CAN_NUM) + model.n_facet_emb + n_facet_local_encoder == n_facet_all

        model.n_facet_all = n_facet_all
        model.n_facet_context = n_facet_context
        model.n_facet_reranker = n_facet_reranker
        model.n_facet_local_encoder = n_facet_local_encoder
        model.n_facet_effective = n_facet
        
        model.n_facet = n_facet #1 in context based GPT2
        model.n_facet_hidden = n_facet_hidden #0
        assert n_facet_MLP <= 0 #-1 or 0
        assert n_facet_window <= 0 # 0
        n_facet_window = - n_facet_window
        n_facet_MLP = - n_facet_MLP
        model.n_facet_MLP = n_facet_MLP
        model.n_facet_window = n_facet_window
        
        model.pointer_mode = pointer_mode
        model.softmax_nonlinear=softmax_nonlinear
        model.context_efficient_mode = context_efficient_mode
        model.reranker_efficient_mode = reranker_efficient_mode
        #model.masking_ratio = masking_ratio
        #model.only_commpute_loss = only_commpute_loss
        #self.starve_idx = -1
        #self.num_vis_bin = 4
        #self.num_vis_bin_loss = 5
        #self.period_idx = -1

        model.reranker_CAN_NUM = reranker_CAN_NUM
        model.context_source = context_source
        model.candidates_from_previous_reranker = candidates_from_previous_reranker

        if n_facet_hidden == 0:
            n_facet_hidden = n_facet

        # for multiple input hidden states
        if n_facet_MLP > 0:
            hidden_state_input_ratio = 1 + n_facet_MLP #1 + 1
            model.MLP_linear = nn.Linear(model.model_dim * (n_facet_hidden * (n_facet_window+1) ), model.model_dim * n_facet_MLP) # (hid_dim*2) -> (hid_dim)
            model.MLP_linear_l2 = nn.Linear(model.model_dim * n_facet_MLP, model.model_dim * n_facet_MLP)
        else:
            hidden_state_input_ratio = n_facet_hidden * (n_facet_window+1) #1 * (0+1)
        print("hidden_state_input_ratio ", hidden_state_input_ratio)
        print("n_facet_all, n_facet_context, n_facet_reranker, n_facet, n_facet_emb, n_facet_local_encoder ", model.n_facet_all, model.n_facet_context, model.n_facet_reranker, model.n_facet, model.n_facet_emb, n_facet_local_encoder)
        print("n_facet_hidden, n_facet_window, n_facet_MLP ", model.n_facet_hidden, model.n_facet_window, model.n_facet_MLP)
        print("Mode: ", model.context_efficient_mode, model.reranker_efficient_mode, model.pointer_mode)
        
        print("model.use_lm_head_context ", model.use_lm_head_context)

        total_lin_dim = model.model_dim * hidden_state_input_ratio
        small_value = 0.0001

        model.project_arr = nn.ModuleList([nn.Linear(total_lin_dim, model.model_dim, bias=use_proj_bias) for i in range(n_facet_all)])
        if n_facet_local_encoder > 0:
            model.encoder_project_arr = nn.ModuleList([nn.Linear(model.model_dim, model.model_dim, bias=use_proj_bias) for i in range(n_facet_local_encoder)])

        #some changes here-----------------------------------------
        #for now, context and reranker will use add or assign at the same time
        for i in range(n_facet_all):
            if use_proj_bias:
                model.project_arr[i].bias.data.zero_()
            linear_weights = torch.zeros_like(model.project_arr[i].weight.data)

            # if i!= n_facet - 1:
            #     linear_weights = linear_weights + small_value * (torch.rand((model.model_dim, total_lin_dim)) - 0.5 )

            #for assign
            if context_efficient_mode.startswith("assign"):
                if i < n_facet + n_facet_context*len(context_source) + n_facet_reranker*len(reranker_CAN_NUM):
                    linear_weights[:,:model.model_dim] = torch.eye(model.model_dim)
                else:
                    linear_weights[:,:model.model_dim] = 1e-10 * torch.eye(model.model_dim)

            elif context_efficient_mode.startswith("add"):
            #for add
#                 if i < n_facet + n_facet_context:
                if i < n_facet:
                    linear_weights[:,:model.model_dim] = torch.eye(model.model_dim)
                else:
                    linear_weights[:,:model.model_dim] = 1e-10 * torch.eye(model.model_dim)
            else:
                print("wrong efficient_mode name")
                assert 1==0
            model.project_arr[i].weight.data = linear_weights

        if len(model.pointer_mode) > 0:
            if model.pointer_mode == 'CopyNet':
                model.init_neg_bias = nn.Parameter( 0 * torch.ones(1) )
            elif model.pointer_mode == 'Pointer':
                model.att_bias = nn.Parameter( torch.zeros( config.hidden_size ) )
                model.weight_gen = nn.Linear(config.hidden_size, 1)
                model.weight_gen.bias.data[:] = 3
                model.att_v = nn.Parameter( torch.zeros( config.hidden_size ) )
            elif model.pointer_mode == 'Pointer_Sentinel':
                model.weight_gen = nn.Linear(config.hidden_size, 1)
                model.weight_gen.bias.data[:] = 0

        model.weight_context = nn.Linear(config.hidden_size * hidden_state_input_ratio, 1)
        model.weight_context.weight.data *= 0.0001
        model.weight_context.bias.data[:] = 0

        if len(weight_mode) > 0:
            model.weight_facet_decoder = nn.Linear(config.hidden_size * hidden_state_input_ratio, model.n_facet_effective)
            #model.weight_facet_decoder = nn.Linear(config.hidden_size * n_facet_hidden * (n_facet_window+1), n_facet)
            model.weight_global = nn.Parameter( torch.ones(model.n_facet_effective) )

        #model.init_weights()
        model.weight_mode = weight_mode
        # model.transformer = transformer_input
        model.vocab_size = config.vocab_size
        # model.d_model = config.d_model
        model.output_probs = True
        #model.starve_idx = -1
        model.c = 100

        # if context_efficient_mode in ['add_average_of_duplicates', 'assign_average_of_duplicates']:
        #     only_before_index = torch.tensor( list(range(seq_len+1)), device=device )
        #     only_before_index = torch.tril(only_before_index.expand(seq_len+1,seq_len+1), diagonal = -1) #lower traingular matrix withou diagonal
        #     self.only_before_index = only_before_index #(seq_len+1 ,seq_len+1)



def dynamic_partition_softmax(decoder_outputs, prev_decoder_hidden_states_all, prev_decoder_input_ids_all, decoder_input_ids, input_ids, encoder_hidden_state, labels, model):
    all_hidden_states = decoder_outputs[2]
    #print(prev_decoder_hidden_states_all)
    #print(prev_decoder_input_ids_all)
    #print(prev_decoder_input_ids_all.shape)
    #print(decoder_input_ids.shape)
    #print(input_ids)
    if labels is None and prev_decoder_hidden_states_all is not None: #ADD Haw-Shiuan
        temp_tuple = tuple()
        for layer, _ in enumerate(prev_decoder_hidden_states_all):
            # print(layer, prev_hidden_states[layer].size(), all_hidden_states[layer].size())
            temp_tuple += (torch.cat((prev_decoder_hidden_states_all[layer], all_hidden_states[layer]), dim=1),)
        #decoder_input_ids = torch.cat((prev_decoder_input_ids_all, decoder_input_ids), dim=1)
        all_hidden_states = temp_tuple
        decoder_input_ids = prev_decoder_input_ids_all
    #print(decoder_input_ids.shape)
    #all_hidden_states -> 13*bsz*seq_len*hidden_size

    #insert extra token to decoder_input_ids
    device = all_hidden_states[0].device

    bsz, seq_len = decoder_input_ids.size()
    temp_col = torch.full((bsz, 1), model.c, dtype = decoder_input_ids.dtype, device=device)
    decoder_input_ids_mod = torch.cat((temp_col, decoder_input_ids), dim=1)

    ## Multi-input hidden states: generate q_ct from hidden states
    #list of hidden state embeddings taken as input
    hidden_emb_arr = []

    # h_  0
    for i in range(model.n_facet_hidden):
        hidden_states = all_hidden_states[-(i+1)] #i-th hidden-state embedding from the top

        # add Rescale based on T5 codes
        if model.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (model.model_dim**-0.5)

        #device = hidden_states.device
        hidden_emb_arr.append(hidden_states)
        for j in range(model.n_facet_window):
            bsz, seq_len, hidden_size = hidden_states.size() #bsz -> , seq_len -> , hidden_size -> 768 in GPT-small?
            if j+1 < hidden_states.size(1):
                shifted_hidden = torch.cat( (torch.zeros( (bsz, (j+1), hidden_size), device = device), hidden_states[:,:-(j+1),:]), dim = 1)
            else:
                shifted_hidden = torch.zeros( (bsz, hidden_states.size(1), hidden_size), device = device)
            hidden_emb_arr.append(shifted_hidden)
    #hidden_emb_arr -> (W*H, bsz, seq_len, hidden_size)

    #n_facet_MLP -> 1
    if model.n_facet_MLP > 0:
        stacked_hidden_emb_raw_arr = torch.cat(hidden_emb_arr, dim=-1) #(bsz, seq_len, W*H*hidden_size)
        # model.MLP_linear = nn.Linear(config.n_embd * (n_facet_hidden * (n_facet_window+1) ), config.n_embd * n_facet_MLP) -> why +1?
        hidden_emb_MLP = model.MLP_linear(stacked_hidden_emb_raw_arr) #bsz, seq_len, hidden_size
        stacked_hidden_emb_arr = torch.cat([hidden_emb_arr[0], gelu(hidden_emb_MLP)], dim=-1) #bsz, seq_len, 2*hidden_size
    else:
        stacked_hidden_emb_arr = hidden_emb_arr[0]

    #list of linear projects per facet
    projected_emb_arr = []
    #list of final logits per facet
    facet_lm_logits_arr = []
    facet_lm_logits_real_arr = []
    #for reranker we need candidate ids
    #topn(lm_head(last_hidden_state), 100) to get top100 candidate_tokens and there lm_logits for every input_id
    rereanker_candidate_token_ids_arr = []

    for i in range(model.n_facet):
    #     #linear projection
        projected_emb = model.get_facet_emb(stacked_hidden_emb_arr, i) #(bsz, seq_len, hidden_dim)
        projected_emb_arr.append(projected_emb)
        #logits for all tokens in vocab
        lm_logits = model.lm_head(projected_emb) #(bsz, seq_len, vocab_size)
        facet_lm_logits_arr.append(lm_logits)
        if i < model.n_facet_reranker and not model.candidates_from_previous_reranker:
            candidate_token_ids = []
            for j in range(len(model.reranker_CAN_NUM)):
                _, candidate_token_ids_ = torch.topk(lm_logits, model.reranker_CAN_NUM[j])
                candidate_token_ids.append(candidate_token_ids_)
            rereanker_candidate_token_ids_arr.append(candidate_token_ids)

    for i in range(model.n_facet_reranker):
        for j in range(len(model.reranker_CAN_NUM)):
            projected_emb = model.get_facet_emb(stacked_hidden_emb_arr, model.n_facet+i*len(model.reranker_CAN_NUM)+j) #(bsz, seq_len, hidden_dim)
            projected_emb_arr.append(projected_emb)

    for i in range(model.n_facet_context):
        for j in range(len(model.context_source)):
            projected_emb = model.get_facet_emb(stacked_hidden_emb_arr, model.n_facet+model.n_facet_reranker*len(model.reranker_CAN_NUM)+i*len(model.context_source)+j) #(bsz, seq_len, hidden_dim)
            projected_emb_arr.append(projected_emb)
    
    for i in range(model.n_facet_local_encoder):
        projected_emb = model.get_facet_emb(stacked_hidden_emb_arr, model.n_facet + model.n_facet_context*len(model.context_source) + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i) #(bsz, seq_len, hidden_dim)
        projected_emb_arr.append(projected_emb)


    #to generate context-based embeddings for words in input
    for i in range(model.n_facet_emb):
        projected_emb = model.get_facet_emb(stacked_hidden_emb_arr, model.n_facet + model.n_facet_context*len(model.context_source) + model.n_facet_reranker*len(model.reranker_CAN_NUM) + model.n_facet_local_encoder + i) #(bsz, seq_len, hidden_dim)
        projected_emb_arr.append(projected_emb)

    if model.use_lm_head_context:
        emb_transform = projected_emb_arr[-1]

    #get weights for tokens in top candidates
    for i in range(model.n_facet_reranker):
        for j in range(len(model.reranker_CAN_NUM)):
            for k in range(bsz):
                bsz, seq_len, hidden_size = projected_emb_arr[i].size()
                if model.candidates_from_previous_reranker:
                    _, candidate_token_ids = torch.topk(facet_lm_logits_arr[i][k], model.reranker_CAN_NUM[j])
                else:
                    candidate_token_ids = rereanker_candidate_token_ids_arr[i][j][k]
                #logit_hidden_reranker_topn = F.linear(projected_emb_arr[model.n_facet + i*len(model.reranker_CAN_NUM) + j][k], model.decoder.weight[candidate_token_ids, :], model.decoder.bias[candidate_token_ids[k,:]])
                logit_hidden_reranker_topn = (projected_emb_arr[model.n_facet + i*len(model.reranker_CAN_NUM) + j][k].unsqueeze(dim=1).expand(seq_len, model.reranker_CAN_NUM[j], hidden_size) * model.lm_head.weight[candidate_token_ids, :]).sum(dim=-1)
                if model.reranker_efficient_mode == 'add_reranker_logits':
                    facet_lm_logits_arr[i][k].scatter_add_(1, candidate_token_ids, logit_hidden_reranker_topn) #(seq_len, vocab_size) <- (seq_len, topk) x (seq_len, topk)
                else:
                    facet_lm_logits_arr[i][k].scatter_(1, candidate_token_ids, logit_hidden_reranker_topn)

    if len(model.pointer_mode) > 0:
        bsz, encoder_seq_len = input_ids.size()
        bsz, seq_len, hidden_size = projected_emb_arr[0].size()
        decoder_hidden = projected_emb_arr[model.n_facet + model.n_facet_context*len(model.context_source) + model.n_facet_reranker*len(model.reranker_CAN_NUM)].unsqueeze(dim=2).expand(bsz, seq_len, encoder_seq_len, hidden_size)
        encoder_hidden = model.encoder_project_arr[i](encoder_hidden_state).unsqueeze(dim=1).expand(bsz, seq_len, encoder_seq_len, hidden_size)
        if model.pointer_mode == 'CopyNet':
            logit = (decoder_hidden * torch.tanh( encoder_hidden ) ).sum(dim=-1) #(bsz, seq_len, encoder_seq_len)
            logit_exp = torch.exp(logit + model.init_neg_bias )

            logit_to_add = logit_exp
        elif model.pointer_mode == 'Pointer':
            logit = torch.sum( model.att_v.expand(bsz, seq_len, encoder_seq_len, hidden_size) * torch.tanh(decoder_hidden + encoder_hidden + model.att_bias.expand(bsz, seq_len, encoder_seq_len, hidden_size) ), dim =-1) #(bsz, seq_len, encoder_seq_len)
            att = torch.softmax(logit, dim = 2)

            logit_to_add = att
        elif model.pointer_mode == 'Pointer_Sentinel':
            gen_weight_logit = model.weight_gen(hidden_emb_arr[0]) #bsz, seq_len, 1
            logit = (decoder_hidden * torch.tanh( encoder_hidden ) ).sum(dim=-1) #(bsz, seq_len, encoder_seq_len)
            logit = torch.cat((logit, gen_weight_logit), dim=-1)
            att = torch.softmax(logit, dim = -1)
            logit_to_add = att[:,:,:-1]
            gen_weight = att[:,:,-1]

        context_logits_to_add = torch.zeros((bsz, seq_len, model.vocab_size), device=device)
        context_logits_to_add.scatter_add_(dim=2, index= input_ids.unsqueeze(dim=1).expand(bsz, seq_len, encoder_seq_len), src=logit_to_add)

    #input_context_partition
    if 'encode' in model.context_source:
        bsz, encoder_seq_len = input_ids.size()
        bsz, seq_len, hidden_size = projected_emb_arr[0].size()
        for i in range(model.n_facet_context):
            # assuming your new context partition is immediatedly after reranker partition with index "model.n_facet + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i"
            logit_hidden_encoder_input = (projected_emb_arr[model.n_facet + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i].unsqueeze(dim=2).expand(bsz, seq_len, encoder_seq_len, hidden_size) * model.lm_head.weight[input_ids, :].unsqueeze(dim=1).expand(bsz, seq_len, encoder_seq_len, hidden_size) ).sum(dim=-1) #(bsz, seq_len, encoder_seq_len)
            
            if model.n_facet_local_encoder > 0:
                logit_hidden_encoder_input += (projected_emb_arr[model.n_facet + model.n_facet_context*len(model.context_source) + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i].unsqueeze(dim=2).expand(bsz, seq_len, encoder_seq_len, hidden_size) * model.encoder_project_arr[i](encoder_hidden_state).unsqueeze(dim=1).expand(bsz, seq_len, encoder_seq_len, hidden_size) ).sum(dim=-1) #(bsz, seq_len, encoder_seq_len)

            # The code below wants to simply achieve "facet_lm_logits_arr[i].scatter_(dim=1, input_ids.unsqueeze(dim=1).expand(bsz, seq_len, encoder_seq_len), logit_hidden_encoder_input)".
            # We cannot directly use one scatter_ command because the warning in https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_ said that the grad would have some problem by doing that.
            context_logits_to_add = torch.zeros((bsz, seq_len, model.vocab_size), device=device)
            context_logits_to_add.scatter_add_(dim=2, index= input_ids.unsqueeze(dim=1).expand(bsz, seq_len, encoder_seq_len), src=logit_hidden_encoder_input)

            if model.context_efficient_mode == 'assign_average_of_duplicates' or model.context_efficient_mode == 'assign_sum_of_duplicates':
                facet_lm_logits_arr[i].scatter_(dim=2, index=input_ids.unsqueeze(dim=1).expand(bsz, seq_len, encoder_seq_len), src=torch.zeros((bsz, seq_len, encoder_seq_len), device=device))

            count_duplicates_before_index = torch.full((bsz, model.vocab_size), 1e-10, requires_grad = False, device=device)
            count_duplicates_before_index.scatter_add_(dim=1, index = input_ids, src=torch.ones((bsz, encoder_seq_len), device = device))

            facet_lm_logits_arr[i] += context_logits_to_add / count_duplicates_before_index.unsqueeze(dim=1).expand(bsz, seq_len, model.vocab_size)
    
    if 'decode' in model.context_source:
        bsz, seq_len, hidden_size = all_hidden_states[-1].size()

        only_before_index = torch.tensor(list(range(seq_len+1)), device=device )
        only_before_index = torch.tril(only_before_index.expand(seq_len+1,seq_len+1), diagonal = -1) #lower traingular matrix withou diagonal
        model.only_before_index = only_before_index #(seq_len+1 ,seq_len+1)

        for i in range(model.n_facet_context):
            bsz, seq_len, hidden_size = projected_emb_arr[i].size()
            if not model.use_lm_head_context:
                logit_hidden_context_arr = []
                for j in range(bsz):
                    if 'encode' in model.context_source:
                        logit_hidden_context_arr.append(F.linear(projected_emb_arr[model.n_facet + model.n_facet_context + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i][j], model.lm_head.weight[decoder_input_ids_mod[j, :], :], None))
                    else:
                        logit_hidden_context_arr.append(F.linear(projected_emb_arr[model.n_facet + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i][j], model.lm_head.weight[decoder_input_ids_mod[j, :], :], None))
                logit_hidden_context = torch.stack(logit_hidden_context_arr, dim =0)
            else:
                logit_hidden_context_arr = []
                for j in range(bsz):
                    if model.n_facet_emb == 1:
                        if 'encode' in model.context_source:
                            logit_hidden_context_arr.append(F.linear(projected_emb_arr[model.n_facet + model.n_facet_context + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i][j], emb_transform[j], None))
                        else:
                            logit_hidden_context_arr.append(F.linear(projected_emb_arr[model.n_facet + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i][j], emb_transform[j], None))
                    elif model.n_facet_emb == 2:
                        if 'encode' in model.context_source:
                            logit = F.linear(projected_emb_arr[model.n_facet + model.n_facet_context + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i][j], model.lm_head.weight[decoder_input_ids[j, :], :], None)
                        else:
                            logit = F.linear(projected_emb_arr[model.n_facet + model.n_facet_reranker*len(model.reranker_CAN_NUM) + i][j], model.lm_head.weight[decoder_input_ids[j, :], :], None)
                        #print(projected_emb_arr[-2].shape)
                        #print(emb_transform.shape)
                        #print(logit.shape)
                        logit += F.linear(projected_emb_arr[-2][j], emb_transform[j], None)
                        logit_hidden_context_arr.append(logit)
                    # logit_hidden_context_arr.append(F.linear(projected_emb_arr[model.n_facet + i][j], emb_transform[j]+model.lm_head.weight[decoder_input_ids[j, :], :], None))
                logit_hidden_context = torch.stack(logit_hidden_context_arr, dim =0) #bsz, seq_len, seq_len
                #temp_col = torch.full((bsz, seq_len, 1), 0.0001, dtype = logit_hidden_context.dtype, device=device)
                temp_col = torch.full((bsz, seq_len, 1), 1000, dtype = logit_hidden_context.dtype, device=device)
                logit_hidden_context = torch.cat((temp_col, logit_hidden_context), dim=2)

            only_before_index_expand = model.only_before_index[1:].unsqueeze(0).expand(bsz, seq_len, seq_len+1)
            only_before_index_expand = only_before_index_expand.to(device=device)
            #count for average, remove gradient calculation
            count_duplicates_before_index = torch.full((bsz, seq_len, model.vocab_size), 0.00000001, requires_grad = False, device=device)
            count_duplicates_before_index.scatter_add_(dim=2, index = decoder_input_ids_mod[:, model.only_before_index[1:]], src=torch.ones((bsz, seq_len, seq_len+1), device = device))
            #get logits before curr word in input_seq
            logit_hidden_context_before = torch.gather(logit_hidden_context, dim=2, index=only_before_index_expand) # (bsz, seq_len, seq_len+1)
            temp = facet_lm_logits_arr[i][:, :, model.c].clone() #(bsz, seq_len)
            #get average of context logits
            context_logits_to_add = torch.zeros((bsz, seq_len, model.vocab_size), device=device)
            context_logits_to_add.scatter_add_(dim=2, index= decoder_input_ids_mod[:, model.only_before_index[1:]], src=logit_hidden_context_before)
            #add
            if model.context_efficient_mode == 'add_average_of_duplicates':
                facet_lm_logits_arr[i] += context_logits_to_add/count_duplicates_before_index
            elif model.context_efficient_mode == 'assign_average_of_duplicates':
                # #assign
                facet_lm_logits_arr[i].scatter_(dim=2, index=decoder_input_ids_mod[:, model.only_before_index[1:]], src=torch.zeros((bsz, seq_len, seq_len+1), device=device))
                facet_lm_logits_arr[i] += context_logits_to_add/count_duplicates_before_index
            # re-assign logits for token model.c in vocab
            facet_lm_logits_arr[i][:, :, model.c] = temp #scatter_add would add to token model.c multiplr times

    #logits for n_facet (==n_facet_effective)
    # for i in range(model.n_facet_context, model.n_facet_all):
    
    facet_lm_logits_real_arr = facet_lm_logits_arr
    #for i in range(model.n_facet):
    #    if model.starve_idx < 0 or model.starve_idx == i:
    #        facet_lm_logits_real_arr.append(facet_lm_logits_arr[i])
    #    else:
    #        facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, model.vocab_size), device=all_hidden_states[-1].device ))

    #with torch.no_grad():
    #    if not model.only_commpute_loss:
    #        stacked_facet_emb = torch.stack(projected_emb_arr, dim=0)
    #        stacked_facet_emb = stacked_facet_emb / (1e-12 + stacked_facet_emb.norm(dim = -1, keepdim=True))
    #        pred_mean = stacked_facet_emb.mean(dim = 0, keepdim = True)
    #        div_raw = (stacked_facet_emb - pred_mean).norm(dim = -1)
    #        emb_div_arr = - div_raw.mean(dim=1).mean(dim=0)
    #        emb_div = emb_div_arr.mean()
        #if vis_seq_loss or vis_simple:
        #    seq_len = emb_div_arr.numel()
        #    num_token_removed = seq_len % model.num_vis_bin
        #    proper_seq_len = seq_len - num_token_removed
        #    emb_div_arr_vis = emb_div_arr[:proper_seq_len].view(model.num_vis_bin,-1).mean(dim=-1)
        #if model.masking_ratio > 0:
        #    num_facet, bsz, seq_len = div_raw.size()
        #    var_avg_flat = div_raw.mean(0).view(-1)
        #    var_avg = var_avg_flat.median()
        #    single_facet_mask = var_avg_flat < var_avg * model.masking_ratio

    # #len(facet_lm_logits_arr) -> n.facet = n_facet_all - n_facet_context

    #weight_mode = ''
    stacked_facet_lm_logits = torch.stack(facet_lm_logits_real_arr, dim=0)
    weight = None
    if model.weight_mode == 'dynamic':
        weight = model.weight_facet_decoder(stacked_hidden_emb_arr).softmax(dim=-1) #hidden_dim*hidden_input_state_ration -> n_facet_effective
    elif model.weight_mode == 'static':
        weight = model.weight_global.softmax(dim=-1) #torch.ones(n_facet_effective)

    #cal stage1 prob
    prediction_prob = 0

    for i in range(model.n_facet_effective):
        facet_lm_logits = facet_lm_logits_real_arr[i]
        if model.pointer_mode == "CopyNet":
            context_logit = torch.log( context_logits_to_add + 1e-20 )
            facet_lm_logits_context = torch.maximum(facet_lm_logits, context_logit)
            facet_lm_logits_sig = torch.exp(facet_lm_logits - facet_lm_logits_context.max(dim=-1,keepdim=True)[0]) + torch.exp( context_logit - facet_lm_logits_context.max(dim=-1,keepdim=True)[0] )
            #print("context_exp_to_add", context_exp_to_add)
            #print("log context_exp_to_add", torch.log( context_exp_to_add + 1e-10 ) )
            #print("exp log context_exp_to_add", torch.exp( torch.log( context_exp_to_add + 1e-10 ) - facet_lm_logits.max(dim=-1,keepdim=True)[0] ) )
            facet_lm_logits_softmax = facet_lm_logits_sig / facet_lm_logits_sig.sum(dim=-1,keepdim=True)
        elif model.pointer_mode == "Pointer":
            gen_weight = torch.sigmoid( model.weight_gen(hidden_emb_arr[0]) )
            facet_lm_logits_softmax = (1-gen_weight) * context_logits_to_add + gen_weight * facet_lm_logits.softmax(dim=-1)
        elif model.pointer_mode == "Pointer_Sentinel":
            facet_lm_logits_softmax = context_logits_to_add + gen_weight.unsqueeze(-1) * facet_lm_logits.softmax(dim=-1)
        
        elif model.softmax_nonlinear == 'sigsoftmax': #'None' here
            facet_lm_logits_sig = torch.exp(facet_lm_logits - facet_lm_logits.max(dim=-1,keepdim=True)[0]) * (1e-20 + torch.sigmoid(facet_lm_logits))
            facet_lm_logits_softmax = facet_lm_logits_sig / facet_lm_logits_sig.sum(dim=-1,keepdim=True)
        elif model.softmax_nonlinear == 'None':
            facet_lm_logits_softmax = facet_lm_logits.softmax(dim=-1) #softmax over final logits
        
        if model.weight_mode == 'dynamic':
            prediction_prob += facet_lm_logits_softmax * weight[:,:,i].unsqueeze(-1)
        elif model.weight_mode == 'static':
            prediction_prob += facet_lm_logits_softmax * weight[i]
        else:
            prediction_prob += facet_lm_logits_softmax / model.n_facet_effective #softmax over final logits/1
        #outputs = (prediction_prob,) + (stacked_facet_lm_logits, ) + decoder_outputs[1:]
        # outputs = (lm_logits,) + decoder_outputs[1:]
    loss = None
    if labels is not None:

        # Shift so that tokens < n predict n
        #shift_labels = labels[..., 1:].contiguous()
        shift_labels = labels.contiguous()

        loss_fct = torch.nn.NLLLoss(reduction='none')
        #shift_prediction_prob = prediction_prob[..., :-1, :].contiguous()
        shift_prediction_prob = prediction_prob.contiguous()
        shift_labels_flat = shift_labels.view(-1)
        loss_raw = loss_fct(torch.log(shift_prediction_prob.view(-1, model.vocab_size)+1e-8), shift_labels_flat)
        loss = loss_raw[shift_labels_flat != -100].mean()

    #logits = facet_lm_logits_real_arr[0]
    logits = torch.log(prediction_prob)
    return logits, loss, all_hidden_states
    
    
    
class T5ForConditionalGeneration_updated(T5ForConditionalGeneration, GenerationMixin_updated):
    def __init__(self, config, n_facet_all = 1, n_facet=1, n_facet_context =0, n_facet_reranker =0, n_facet_local_encoder=0,
                 n_facet_hidden=1, n_facet_window=0, n_facet_MLP=0, use_proj_bias=False, weight_mode = '',
                 softmax_nonlinear='None', context_efficient_mode='assign_average_of_duplicates',
                 reranker_efficient_mode='assign_only_last_of_duplicates',
                context_source=['encode', 'decode'], reranker_CAN_NUM=[20], reranker_pred='direction', candidates_from_previous_reranker=True, pointer_mode=''):

        super().__init__(config)

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        #---------------------------------------------------------------------
        dynamic_partition_softmax_init(self, config, n_facet_all, n_facet, n_facet_context, n_facet_reranker, n_facet_local_encoder, n_facet_hidden, n_facet_window, n_facet_MLP, use_proj_bias,
                 weight_mode, softmax_nonlinear, context_efficient_mode, reranker_efficient_mode,
                 context_source, reranker_CAN_NUM, candidates_from_previous_reranker, pointer_mode)



    def get_facet_emb(self,input_emb, i):
        return self.project_arr[i](input_emb)

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def wte(self):
        """
        Get the weights for the token embeddings from the transformer
        """
        return self.shared

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        non_tensor__tags=None, # columns from excell # used in the trainer
        non_tensor__raw_inputs=None, # columns from excell # used in the trainer
        #vis_simple=False,
        #vis_seq_loss = False,
        prev_decoder_hidden_states_all = None, #ADD Haw-Shiuan
        prev_decoder_input_ids_all = None #ADD Haw-Shiuan
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        global debug

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]


        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        #one is completed decoder_input_ids till now
        #one is previous hidden_state

        #for generation time, check decoder_input_ids and input_ids for context partition

        #----------------------------------------------change to multi-facet version

        logits, loss, all_hidden_states = dynamic_partition_softmax(decoder_outputs, prev_decoder_hidden_states_all, prev_decoder_input_ids_all, decoder_input_ids, input_ids, hidden_states, labels, self)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_hidden_states_all=all_hidden_states, #ADD Haw-Shiuan
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_outputs=encoder_outputs,
        )