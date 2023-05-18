import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
import random
from torch_scatter import scatter_mul

from .configuration_gpt2 import GPT2Config
from .file_utils import add_start_docstrings
from .modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer
from .modeling_gpt2_multi import GPT2PreTrainedModel


logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
    "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-pytorch_model.bin",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin",
}

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class GPT2RerankHeadLMModel(GPT2PreTrainedModel):
    def __init__(self, config, transformer_input, n_facet_all = 2, n_facet=1, n_facet_context =1, n_facet_reranker =1, 
                 n_facet_stage2 =0, n_facet_hidden=1, n_facet_window=0, n_facet_MLP=0, use_proj_bias=False, weight_mode = '', 
                 only_commpute_loss=False, softmax_nonlinear='None', context_efficient_mode='None', 
                 reranker_efficient_mode='None',  reranker_stage2_efficient_mode='None',
                 masking_ratio=-1, seq_len = 0, device=None, n_facet_effective_in=-1, last_num=0, use_lm_head_context = False, 
                 stage2_CAN_NUM=100, reranker_CAN_NUM=[200, 20], reranker_pred='direction', 
                 candidates_from_previous_reranker=True, stage2_Block_size=10, pointer_mode = ""):

        super(GPT2RerankHeadLMModel, self).__init__(config)
        
        assert n_facet_context <= n_facet
        self.use_lm_head_context = use_lm_head_context
        if self.use_lm_head_context:
            #self.n_facet_emb = 1
            self.n_facet_emb = 2
        else:
            self.n_facet_emb = 0
        print(n_facet_all, n_facet, n_facet_context, n_facet_reranker, n_facet_stage2, self.n_facet_emb)
        assert n_facet + n_facet_context + n_facet_reranker*len(reranker_CAN_NUM) + n_facet_stage2 + self.n_facet_emb == n_facet_all

        self.n_facet_all = n_facet_all
        self.n_facet_context = n_facet_context
        self.n_facet_reranker = n_facet_reranker
        self.n_facet_stage2 = n_facet_stage2
        self.n_facet_effective = n_facet

        self.n_facet = n_facet #1 in context based GPT2
        self.n_facet_hidden = n_facet_hidden #0
        assert n_facet_MLP <= 0 #-1 or 0
        assert n_facet_window <= 0 # 0
        n_facet_window = - n_facet_window
        n_facet_MLP = - n_facet_MLP
        self.n_facet_MLP = n_facet_MLP
        self.n_facet_window = n_facet_window

        self.softmax_nonlinear=softmax_nonlinear
        self.context_efficient_mode = context_efficient_mode
        self.reranker_efficient_mode = reranker_efficient_mode
        self.reranker_stage2_efficient_mode = reranker_stage2_efficient_mode
        self.masking_ratio = masking_ratio
        self.only_commpute_loss = only_commpute_loss
        self.starve_idx = -1
        
        self.num_vis_bin = 4
        self.num_vis_bin_loss = 5
        self.period_idx = -1
        
        self.pointer_mode = pointer_mode

        self.stage2_CAN_NUM = stage2_CAN_NUM
        self.reranker_CAN_NUM = reranker_CAN_NUM
        self.candidates_from_previous_reranker=candidates_from_previous_reranker
        self.reranker_pred=reranker_pred
        
        if n_facet_hidden == 0:
            n_facet_hidden = n_facet
        
        # for multiple input hidden states
        if n_facet_MLP > 0:
            hidden_state_input_ratio = 1 + n_facet_MLP #1 + 1
            self.MLP_linear = nn.Linear(config.n_embd * (n_facet_hidden * (n_facet_window+1) ), config.n_embd * n_facet_MLP) # (hid_dim*2) -> (hid_dim)
            #self.MLP_linear_l2 = nn.Linear(config.n_embd * n_facet_MLP, config.n_embd * n_facet_MLP)
        else:            
            hidden_state_input_ratio = n_facet_hidden * (n_facet_window+1) #1 * (0+1)
        print("hidden_state_input_ratio ", hidden_state_input_ratio)
        print("n_facet_all, n_facet_context, n_facet_reranker, n_facet_stage2, n_facet, n_facet_emb ", self.n_facet_all, self.n_facet_context, self.n_facet_reranker, self.n_facet_stage2, self.n_facet, self.n_facet_emb)
        print("n_facet_hidden, n_facet_window, n_facet_MLP ", self.n_facet_hidden, self.n_facet_window, self.n_facet_MLP)
        print("Mode: ", self.context_efficient_mode, self.reranker_efficient_mode, self.reranker_stage2_efficient_mode)
        print("self.use_lm_head_context ", self.use_lm_head_context)

        total_lin_dim = config.n_embd * hidden_state_input_ratio
        #small_value = 0.0001
        if len(self.pointer_mode) > 0:
            small_value = 1e-10
        else:
            small_value = 1e-10
        
        self.project_arr = nn.ModuleList([nn.Linear(total_lin_dim, config.n_embd, bias=use_proj_bias) for i in range(n_facet_all)])
        
        #some changes here-----------------------------------------
        #for now, context and reranker will use add or assign at the same time
        for i in range(n_facet_all):
            if use_proj_bias:
                self.project_arr[i].bias.data.zero_()
            linear_weights = torch.zeros_like(self.project_arr[i].weight.data)

            #for assign
            if context_efficient_mode.startswith("assign"):
                if i < n_facet + n_facet_context + n_facet_reranker*len(self.reranker_CAN_NUM):
                    linear_weights[:,:config.n_embd] = torch.eye(config.n_embd)
                else:
                    linear_weights[:,:config.n_embd] = small_value * torch.eye(config.n_embd)
                    
            elif context_efficient_mode.startswith("add"):
            #for add
#                 if i < n_facet + n_facet_context:
                if i < n_facet:
                    linear_weights[:,:config.n_embd] = torch.eye(config.n_embd)
                else:
                    linear_weights[:,:config.n_embd] = small_value * torch.eye(config.n_embd)
            else:
                print("wrong efficient_mode name")
                assert 1==0
            self.project_arr[i].weight.data = linear_weights

        #self.project_emb = nn.Linear(config.n_embd, config.n_embd, bias=use_proj_bias)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        if self.reranker_pred == 'magnitude':
            self.rerank_magnitude_head = nn.ModuleList([nn.Linear(config.n_embd, 1, bias=True) for i in range(len(self.reranker_CAN_NUM))])
            for i in range(len(self.reranker_CAN_NUM)):
                self.rerank_magnitude_head[i].weight.data.normal_(mean=0.0, std=0.000002)
                self.rerank_magnitude_head[i].bias.data[:] = 1
        elif self.reranker_pred == 'zoom_mul':
            self.rerank_magnitude_head = nn.ModuleList([nn.Linear(config.n_embd, self.reranker_CAN_NUM[i], bias=True) for i in range(len(self.reranker_CAN_NUM))])
            for i in range(len(self.reranker_CAN_NUM)):
                self.rerank_magnitude_head[i].weight.data.normal_(mean=0.0, std=0.000002)
                self.rerank_magnitude_head[i].bias.data[:] = 1
        elif self.reranker_pred == 'zoom_add':
            self.rerank_magnitude_head = nn.ModuleList([nn.Linear(config.n_embd, self.reranker_CAN_NUM[i], bias=False) for i in range(len(self.reranker_CAN_NUM))])
            for i in range(len(self.reranker_CAN_NUM)):
                self.rerank_magnitude_head[i].weight.data.normal_(mean=0.0, std=0.000002)

         
        #if context_efficient_mode in ['add_average_of_duplicates', 'assign_average_of_duplicates']:
        if len(context_efficient_mode) > 0:
            only_before_index = torch.tensor( list(range(seq_len+1)), device=device )
            only_before_index = torch.tril(only_before_index.expand(seq_len+1,seq_len+1), diagonal = -1) #lower traingular matrix withou diagonal
            self.only_before_index = only_before_index #(seq_len+1 ,seq_len+1)


        #self.weight_context = nn.Linear(config.hidden_size * hidden_state_input_ratio, 1)
        #self.weight_context.weight.data *= 0.0001 
        #self.weight_context.bias.data[:] = 0
        

        if len(weight_mode) > 0:
            if weight_mode == 'dynamic':
                self.weight_facet_decoder = nn.Linear(config.hidden_size * hidden_state_input_ratio, self.n_facet_effective)
            elif weight_mode == 'static':
            #self.weight_facet_decoder = nn.Linear(config.hidden_size * n_facet_hidden * (n_facet_window+1), n_facet)
                self.weight_global = nn.Parameter( torch.ones(self.n_facet_effective) )
        
        if len(self.pointer_mode) > 0:
            assert self.n_facet_emb == 2
            #assert self.n_facet_all == 3
            if self.pointer_mode == 'CopyNet' or self.pointer_mode == 'cache':
                #self.init_neg_bias = nn.Parameter( -1 * torch.ones(1) )
                #self.init_neg_bias = nn.Parameter( -1e1 * torch.ones(1) )
                self.init_neg_bias = nn.Parameter( -100 * torch.ones(1) )
                #self.init_neg_bias = nn.Parameter( -1e2 * torch.ones(1) )
                #self.init_neg_bias = nn.Parameter( -1e3 * torch.ones(1) )
                #self.init_neg_bias = nn.Parameter( -1e10 * torch.ones(1) )
            elif self.pointer_mode == 'Pointer':
                self.att_bias = nn.Parameter( torch.zeros( config.hidden_size ) )
                self.weight_gen = nn.Linear(config.hidden_size, 1)
                self.weight_gen.bias.data[:] = 0
                #self.ptr_bias = nn.Parameter( 1e5 * torch.zeros(1) )
                self.att_v = nn.Parameter( torch.zeros( config.hidden_size ) )
            elif self.pointer_mode == 'Pointer_Sentinel':
                #self.att_bias = nn.Parameter( torch.zeros( config.hidden_size ) )
                self.weight_gen = nn.Linear(config.hidden_size, 1)
                self.weight_gen.bias.data[:] = 0
                #self.ptr_bias = nn.Parameter( 1e5 * torch.zeros(1) )

        #self.init_weights()
        self.weight_mode = weight_mode
        self.transformer = transformer_input
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.output_probs = True
        self.starve_idx = -1
        self.c = 100
        
        self.Block_size = stage2_Block_size
        self.num_of_rerank = int(seq_len/self.Block_size)
        self.rerank_places = np.arange(0, seq_len, self.Block_size)
        print("Block_size:", self.Block_size)
        print("num_of_rerank:", self.num_of_rerank)
        print("rerank_places:", self.rerank_places)
        #num_of_rerank * Block_size == all the tokens to be do reranking stage2
        #idealy it should be equal to bptt

        if self.n_facet_stage2>0:
            self.rerank_linear_head = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.stage1H_linear_head = nn.Linear(config.n_embd, config.n_embd, bias=False)

            #self.init_weights()
            self.rerank_linear_head.weight.data.normal_(mean=0.0, std=0.0002)
            self.stage1H_linear_head.weight.data.normal_(mean=0.0, std=0.0002)

    def get_facet_emb(self,input_emb, i):
        return self.project_arr[i](input_emb)

    def get_output_embeddings(self):
        return self.lm_head
    
    @property
    def wte(self):
        """
        Get the weights for the token embeddings from the transformer
        """
        return self.transformer.wte

    # def prepare_inputs_for_generation(self, input_ids, **kwargs):
    #     # only last token for inputs_ids if past is defined in kwargs
    #     if "past" in kwargs and kwargs["past"]:
    #         input_ids = input_ids[:, -1].unsqueeze(-1)
    #     inputs = {"input_ids": input_ids}
    #     inputs.update(kwargs)
    #     return inputs

    def pointer_network_scatter(self, logit_exp_hidden_context, input_ids_mod, device):
        bsz, seq_len, seq_len_plus_one = logit_exp_hidden_context.size()
        only_before_index_expand = self.only_before_index[1:seq_len+1,:seq_len+1].unsqueeze(0).expand(bsz, seq_len, seq_len+1).to(device=device)
        logit_exp_hidden_context_before = torch.gather(logit_exp_hidden_context, dim=2, index=only_before_index_expand) # (bsz, seq_len, seq_len+1)
        context_exp_to_add = torch.zeros((bsz, seq_len, self.vocab_size), device=device)
        context_exp_to_add.scatter_add_(dim=2, index= input_ids_mod[:, self.only_before_index[1:seq_len+1,:seq_len+1]], src=logit_exp_hidden_context_before)
        return context_exp_to_add

    def prepare_only_before_index_expand(self, bsz, seq_len, device):
        only_before_index_expand = self.only_before_index[1:seq_len+1,:seq_len+1].unsqueeze(0).expand(bsz, seq_len, seq_len+1)
        only_before_index_expand = only_before_index_expand.to(device=device)
        return only_before_index_expand
    
    def prepare_count_duplicates_before_index(self, input_ids_mod, device):
        bsz, seq_len_plus_one = input_ids_mod.size()
        seq_len = seq_len_plus_one - 1
        count_duplicates_before_index = torch.full((bsz, seq_len, self.vocab_size), 0.00000001, requires_grad = False, device=device)
        count_duplicates_before_index.scatter_add_(dim=2, index = input_ids_mod[:, self.only_before_index[1:seq_len+1,:seq_len+1]], src=torch.ones((bsz, seq_len, seq_len+1), device = device))
        return count_duplicates_before_index

    def scatter_logits(self, logit_hidden_context, input_ids_mod, facet_lm_logits_arr, only_before_index_expand, count_duplicates_before_index, i, device):
        #bsz, seq_len, hidden_size = projected_emb_arr[i].size()
        bsz, seq_len, vocab_size = facet_lm_logits_arr[i].size()
        #count for average, remove gradient calculation
        #get logits before curr word in input_seq
        logit_hidden_context_before = torch.gather(logit_hidden_context, dim=2, index=only_before_index_expand) # (bsz, seq_len, seq_len+1)
        temp = facet_lm_logits_arr[i][:, :, self.c].clone() #(bsz, seq_len)
        #get average of context logits
        context_logits_to_add = torch.zeros((bsz, seq_len, self.vocab_size), device=device)
        context_logits_to_add.scatter_add_(dim=2, index= input_ids_mod[:, self.only_before_index[1:seq_len+1,:seq_len+1]], src=logit_hidden_context_before)                
        if self.context_efficient_mode == 'assign_average_of_duplicates' or self.context_efficient_mode == 'assign_sum_of_duplicates':
            facet_lm_logits_arr[i].scatter_(dim=2, index=input_ids_mod[:, self.only_before_index[1:seq_len+1,:seq_len+1]], src=torch.zeros((bsz, seq_len, seq_len+1), device=device))
        if count_duplicates_before_index is not None:
            facet_lm_logits_arr[i] += context_logits_to_add / count_duplicates_before_index
        else:
            facet_lm_logits_arr[i] += context_logits_to_add
        #add
        #if self.context_efficient_mode == 'add_average_of_duplicates':
        #    facet_lm_logits_arr[i] += context_logits_to_add / count_duplicates_before_index
        #elif self.context_efficient_mode == 'assign_average_of_duplicates':
        #    facet_lm_logits_arr[i].scatter_(dim=2, index=input_ids_mod[:, self.only_before_index[1:]], src=torch.zeros((bsz, seq_len, seq_len+1), device=device))
        #    facet_lm_logits_arr[i] += context_logits_to_add / count_duplicates_before_index
        #elif self.context_efficient_mode == 'add_sum_of_duplicates':
        #    facet_lm_logits_arr[i] += context_logits_to_add
        #elif self.context_efficient_mode == 'assign_sum_of_duplicates':
        #    facet_lm_logits_arr[i].scatter_(dim=2, index=input_ids_mod[:, self.only_before_index[1:]], src=torch.zeros((bsz, seq_len, seq_len+1), device=device))
        #    facet_lm_logits_arr[i] += context_logits_to_add
        # re-assign logits for token self.c in vocab
        facet_lm_logits_arr[i][:, :, self.c] = temp #scatter_add would add to token self.c multiplr times


    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, output_weight=True, eval_recon_top_k=None, 
                vis_simple=False, vis_seq_loss = False, exclude_neg_labels_for_MRR=False,
                all_input_ids = None, prev_hidden_states = None):
       
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids = token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        all_hidden_states = transformer_outputs[2]
        #all_hidden_states -> 13*bsz*seq_len*hidden_size
        
        if labels is None and prev_hidden_states is not None:
            temp_tuple = tuple()
            for layer, _ in enumerate(prev_hidden_states):
                # print(layer, prev_hidden_states[layer].size(), all_hidden_states[layer].size())
                temp_tuple += (torch.cat((prev_hidden_states[layer], all_hidden_states[layer]), dim=1),)
            input_ids = torch.cat((all_input_ids, input_ids), dim=1)
            all_hidden_states = temp_tuple
        
        #insert extra token to input_ids
        device = all_hidden_states[0].device
        bsz, seq_len = input_ids.size()
        temp_col = torch.full((bsz, 1), self.c, dtype = input_ids.dtype, device=device)
        input_ids_mod = torch.cat((temp_col, input_ids), dim=1)

        ## Multi-input hidden states: generate q_ct from hidden states
        #list of hidden state embeddings taken as input
        hidden_emb_arr = []

        # h_  0
        for i in range(self.n_facet_hidden):
            hidden_states = all_hidden_states[-(i+1)] #i-th hidden-state embedding from the top
            device = hidden_states.device
            hidden_emb_arr.append(hidden_states)
            for j in range(self.n_facet_window):
                bsz, seq_len, hidden_size = hidden_states.size() #bsz -> , seq_len -> , hidden_size -> 768 in GPT-small?
                if j+1 < hidden_states.size(1):
                    shifted_hidden = torch.cat( (torch.zeros( (bsz, (j+1), hidden_size), device = device), hidden_states[:,:-(j+1),:]), dim = 1)
                else:
                    shifted_hidden = torch.zeros( (bsz, hidden_states.size(1), hidden_size), device = device)
                hidden_emb_arr.append(shifted_hidden)
        #hidden_emb_arr -> (W*H, bsz, seq_len, hidden_size)

        #n_facet_MLP -> 1
        if self.n_facet_MLP > 0:
            stacked_hidden_emb_raw_arr = torch.cat(hidden_emb_arr, dim=-1) #(bsz, seq_len, W*H*hidden_size)
            # self.MLP_linear = nn.Linear(config.n_embd * (n_facet_hidden * (n_facet_window+1) ), config.n_embd * n_facet_MLP) -> why +1?
            hidden_emb_MLP = self.MLP_linear(stacked_hidden_emb_raw_arr) #bsz, seq_len, hidden_size
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
        
        for i in range(self.n_facet):
        #     #linear projection
            projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, i) #(bsz, seq_len, hidden_dim)
            projected_emb_arr.append(projected_emb) 
            #logits for all tokens in vocab
            lm_logits = self.lm_head(projected_emb) #(bsz, seq_len, vocab_size)
            facet_lm_logits_arr.append(lm_logits)
            if i < self.n_facet_reranker and not self.candidates_from_previous_reranker:
                candidate_token_ids = []
                for j in range(len(self.reranker_CAN_NUM)):
                    _, candidate_token_ids_ = torch.topk(lm_logits, self.reranker_CAN_NUM[j])
                    candidate_token_ids.append(candidate_token_ids_)
                rereanker_candidate_token_ids_arr.append(candidate_token_ids)
                    
        for i in range(self.n_facet_reranker):
            for j in range(len(self.reranker_CAN_NUM)):
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, self.n_facet+i*len(self.reranker_CAN_NUM)+j) #(bsz, seq_len, hidden_dim)
                projected_emb_arr.append(projected_emb)
            
        for i in range(self.n_facet_context):
            projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, self.n_facet+self.n_facet_reranker*len(self.reranker_CAN_NUM)+i) #(bsz, seq_len, hidden_dim)
            projected_emb_arr.append(projected_emb)
            
        for i in range(self.n_facet_stage2):
            projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, self.n_facet+self.n_facet_reranker*len(self.reranker_CAN_NUM)+self.n_facet_context+i) #(bsz, seq_len, hidden_dim)
            projected_emb_arr.append(projected_emb)

        #to generate context-based embeddings for words in input
        for i in range(self.n_facet_emb):
            projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + self.n_facet_context + self.n_facet_stage2 + i) #(bsz, seq_len, hidden_dim)
            projected_emb_arr.append(projected_emb)

        if self.use_lm_head_context:
        #    emb_transform = self.project_emb(projected_emb_arr[-1])
            emb_transform = projected_emb_arr[-1]
        
        #get weights for tokens in top candidates  
        if self.reranker_pred in ['zoom_mul', 'zoom_add', 'magnitude', 'direction']:
            for i in range(self.n_facet_reranker):
                for j in range(len(self.reranker_CAN_NUM)):
                    if self.reranker_pred == 'direction':
                        bsz, seq_len, hidden_size = projected_emb_arr[i].size()
                        if self.candidates_from_previous_reranker:
                            _, candidate_token_ids = torch.topk(facet_lm_logits_arr[i], self.reranker_CAN_NUM[j])
                        else:
                            candidate_token_ids = rereanker_candidate_token_ids_arr[i][j]
                        logit_hidden_reranker_topn = (projected_emb_arr[self.n_facet + i*len(self.reranker_CAN_NUM) + j].unsqueeze(dim=2).expand(bsz, seq_len, self.reranker_CAN_NUM[j], hidden_size) * self.lm_head.weight[candidate_token_ids, :]).sum(dim=-1)
                        if self.reranker_efficient_mode == 'add_reranker_logits':
                            facet_lm_logits_arr[i].scatter_add_(2, candidate_token_ids, logit_hidden_reranker_topn) #(seq_len, vocab_size) <- (seq_len, topk) x (seq_len, topk)
                        else:
                            facet_lm_logits_arr[i].scatter_(2, candidate_token_ids, logit_hidden_reranker_topn)
                    elif self.reranker_pred == 'zoom_mul' or self.reranker_pred == 'zoom_add':
                        reranker_magnitude_scale = self.rerank_magnitude_head[j](projected_emb_arr[self.n_facet + i*len(self.reranker_CAN_NUM) + j])
                        if self.candidates_from_previous_reranker:
                            _, candidate_token_ids = torch.topk(facet_lm_logits_arr[i], self.reranker_CAN_NUM[j]) 
                        else:
                            candidate_token_ids = rereanker_candidate_token_ids_arr[i][j]
                        if self.reranker_pred == 'zoom_mul':
                            facet_lm_logits_arr[i] = scatter_mul(reranker_magnitude_scale, candidate_token_ids, out=facet_lm_logits_arr[i].clone())
                        else:
                            facet_lm_logits_arr[i] = facet_lm_logits_arr[i].scatter_add(2, candidate_token_ids, reranker_magnitude_scale)    
                    else:#self.reranker_pred == 'magnitude':
                        reranker_magnitude_scale = self.rerank_magnitude_head[j](projected_emb_arr[self.n_facet + i*len(self.reranker_CAN_NUM) + j])
                        logit_hidden_reranker = self.lm_head(projected_emb_arr[i] * reranker_magnitude_scale)
                        if self.candidates_from_previous_reranker:
                            _, candidate_token_ids = torch.topk(facet_lm_logits_arr[i], self.reranker_CAN_NUM[j])
                            logit_hidden_reranker_topn = logit_hidden_reranker.gather(2, candidate_token_ids)
                            if self.reranker_efficient_mode == 'add_reranker_logits':
                                facet_lm_logits_arr[i] = facet_lm_logits_arr[i].scatter_add(2, candidate_token_ids, logit_hidden_reranker_topn)
                            else:
                                facet_lm_logits_arr[i] = facet_lm_logits_arr[i].scatter(2, candidate_token_ids, logit_hidden_reranker_topn)
                        else:
                            logit_hidden_reranker_topn = logit_hidden_reranker.gather(2, rereanker_candidate_token_ids_arr[i][j])
                            if self.reranker_efficient_mode == 'add_reranker_logits':
                                facet_lm_logits_arr[i] = facet_lm_logits_arr[i].scatter_add(2, rereanker_candidate_token_ids_arr[i][j], logit_hidden_reranker_topn)
                            else:
                                facet_lm_logits_arr[i] = facet_lm_logits_arr[i].scatter(2, rereanker_candidate_token_ids_arr[i][j], logit_hidden_reranker_topn)

  
        

        #if self.context_efficient_mode in ['add_average_of_duplicates', 'assign_average_of_duplicates']:
        if len(self.context_efficient_mode) > 0:
            bsz, seq_len, hidden_size = all_hidden_states[-1].size()       
            only_before_index_expand = None
            count_duplicates_before_index = None
            if self.n_facet_emb > 0:
                assert self.n_facet_emb == 2
                if len(self.pointer_mode) == 0 and self.n_facet_context == 0:
                    #logit_hidden_context_arr = []
                    #for j in range(bsz):
                    #    logit = F.linear(projected_emb_arr[-2][j], emb_transform[j], None)
                    #    logit_hidden_context_arr.append(logit)
                    #logit_hidden_context = torch.stack(logit_hidden_context_arr, dim =0)
                    logit_hidden_context = (projected_emb_arr[-2].unsqueeze(dim=2).expand(bsz,seq_len,seq_len,hidden_size) * emb_transform.unsqueeze(dim=1).expand(bsz,seq_len,seq_len,hidden_size) ).sum(-1)
                    temp_col = torch.full((bsz, seq_len, 1), 1000, dtype = logit_hidden_context.dtype, device=device)
                    logit_hidden_context = torch.cat((temp_col, logit_hidden_context), dim=2)
                    
                    if 'average_of_duplicates' in self.context_efficient_mode:
                        count_duplicates_before_index = self.prepare_count_duplicates_before_index(input_ids_mod, device)
                    only_before_index_expand =  self.prepare_only_before_index_expand(bsz, seq_len, device)
                    self.scatter_logits(logit_hidden_context, input_ids_mod, facet_lm_logits_arr, only_before_index_expand, count_duplicates_before_index, 0, device)
                elif self.pointer_mode == 'CopyNet' or self.pointer_mode == 'cache':
                    #logit_exp_hidden_context_arr = []
                    #for j in range(bsz):
                    #    if self.pointer_mode == 'CopyNet':
                    #        logit_exp = torch.exp( F.linear(projected_emb_arr[-2][j], torch.tanh(emb_transform[j]), None) + self.init_neg_bias )
                    #    else:
                    #        logit_exp = torch.exp( F.linear(projected_emb_arr[-2][j], emb_transform[j], None) + self.init_neg_bias )
                    #    logit_exp_hidden_context_arr.append(logit_exp)
                    #logit_exp_hidden_context = torch.stack(logit_exp_hidden_context_arr, dim =0)
                    if self.pointer_mode == 'CopyNet':
                        logit_exp_hidden_context = torch.exp( (projected_emb_arr[-2].unsqueeze(dim=2).expand(bsz,seq_len,seq_len,hidden_size) * torch.tanh(emb_transform.unsqueeze(dim=1).expand(bsz,seq_len,seq_len,hidden_size))).sum(dim=-1) )
                    else:
                        logit_exp_hidden_context = torch.exp( (projected_emb_arr[-2].unsqueeze(dim=2).expand(bsz,seq_len,seq_len,hidden_size) * emb_transform.unsqueeze(dim=1).expand(bsz,seq_len,seq_len,hidden_size)).sum(dim=-1) )
                    #print(logit_exp_hidden_context)
                    temp_col = torch.full((bsz, seq_len, 1), 0, dtype = logit_exp_hidden_context.dtype, device=device)
                    logit_exp_hidden_context = torch.cat((temp_col, logit_exp_hidden_context), dim=2)
                    context_exp_to_add = self.pointer_network_scatter(logit_exp_hidden_context, input_ids_mod, device)
                elif self.pointer_mode == 'Pointer':
                    #att_hidden_context_arr = []
                    #for j in range(bsz):
                    #    logit = torch.sum( self.att_v.expand(seq_len, seq_len, hidden_size) * torch.tanh(projected_emb_arr[-2][j].unsqueeze(dim=1).expand(seq_len, seq_len, hidden_size) + emb_transform[j].unsqueeze(dim=0).expand(seq_len, seq_len, hidden_size) + self.att_bias.expand(seq_len, seq_len, hidden_size) ), dim =2)
                    #    att = torch.softmax(logit, dim = 1)
                    #    att_hidden_context_arr.append(att)
                    #att_hidden_context = torch.stack(att_hidden_context_arr, dim =0)
                    logit = torch.sum( self.att_v.expand(bsz, seq_len, seq_len, hidden_size) * torch.tanh(projected_emb_arr[-2].unsqueeze(dim=2).expand(bsz, seq_len, seq_len, hidden_size) + emb_transform.unsqueeze(dim=1).expand(bsz, seq_len, seq_len, hidden_size) + self.att_bias.expand(bsz, seq_len, seq_len, hidden_size) ), dim =-1)
                    att_hidden_context = torch.softmax(logit, dim = 2)

                    temp_col = torch.full((bsz, seq_len, 1), 0, dtype = att_hidden_context.dtype, device=device)
                    att_hidden_context = torch.cat((temp_col, att_hidden_context), dim=2)
                    prob_to_add = self.pointer_network_scatter(att_hidden_context, input_ids_mod, device)
                elif self.pointer_mode == 'Pointer_Sentinel':
                    gen_weight_logit = self.weight_gen(hidden_emb_arr[0]) #bsz, seq_len, 1

                    logit = (projected_emb_arr[-2].unsqueeze(dim=2).expand(bsz,seq_len,seq_len,hidden_size) * torch.tanh( emb_transform.unsqueeze(dim=1).expand(bsz,seq_len,seq_len,hidden_size) ) ).sum(dim=-1) #(bsz, seq_len, encoder_seq_len)
                    logit = torch.cat((logit, gen_weight_logit), dim=-1)
                    att = torch.softmax(logit, dim = -1)
                    att_hidden_context = att[:,:,:-1]
                    gen_weight = att[:,:,-1].unsqueeze(-1)

                    #att_hidden_context_arr = []
                    #gen_weight_arr = []
                    #for j in range(bsz):
                    #    logit = F.linear( torch.tanh(projected_emb_arr[-2][j]), emb_transform[j], None) #seq_len, 1
                    #    #logit = F.linear( projected_emb_arr[-2][j], emb_transform[j], None) #seq_len, 1
                    #    logit = torch.cat((logit, gen_weight_logit[j]), dim=1)
                    #    att = torch.softmax(logit, dim = 1)
                    #    att_hidden_context_arr.append(att[:,:-1])
                    #    gen_weight_arr.append(att[:,-1].unsqueeze(-1))
                    #att_hidden_context = torch.stack(att_hidden_context_arr, dim =0)
                    #gen_weight = torch.stack(gen_weight_arr, dim =0)
                    temp_col = torch.full((bsz, seq_len, 1), 0, dtype = att_hidden_context.dtype, device=device)
                    att_hidden_context = torch.cat((temp_col, att_hidden_context), dim=2)
                    prob_to_add = self.pointer_network_scatter(att_hidden_context, input_ids_mod, device)
            
            if self.n_facet_context > 0:
                if count_duplicates_before_index is None:
                    count_duplicates_before_index = self.prepare_count_duplicates_before_index(input_ids_mod, device)
                if only_before_index_expand is None:
                    only_before_index_expand =  self.prepare_only_before_index_expand(bsz, seq_len, device)
                for i in range(self.n_facet_context):
                    #bsz, seq_len, hidden_size = projected_emb_arr[i].size()
                    #if not self.use_lm_head_context:
                    #    logit_hidden_context_arr = []
                    #    for j in range(bsz):
                    #        logit_hidden_context_arr.append(F.linear(projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i][j], self.lm_head.weight[input_ids_mod[j, :], :], None))
                    #    logit_hidden_context = torch.stack(logit_hidden_context_arr, dim =0)
                    #else:
                    #    logit_hidden_context_arr = []
                    #    for j in range(bsz):
                    #        if self.n_facet_emb == 1:
                    #            logit_hidden_context_arr.append(F.linear(projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i][j], emb_transform[j], None))
                    #        elif self.n_facet_emb == 2:
                    #            logit = F.linear(projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i][j], self.lm_head.weight[input_ids[j, :], :], None)
                    #            logit += F.linear(projected_emb_arr[-2][j], emb_transform[j], None)
                    #            logit_hidden_context_arr.append(logit)
                    #        # logit_hidden_context_arr.append(F.linear(projected_emb_arr[self.n_facet + i][j], emb_transform[j]+self.lm_head.weight[input_ids[j, :], :], None))
                    #    logit_hidden_context = torch.stack(logit_hidden_context_arr, dim =0) #bsz, seq_len, seq_len
                    #    #temp_col = torch.full((bsz, seq_len, 1), 0.0001, dtype = logit_hidden_context.dtype, device=device)
                    #    temp_col = torch.full((bsz, seq_len, 1), 1000, dtype = logit_hidden_context.dtype, device=device)
                    #    logit_hidden_context = torch.cat((temp_col, logit_hidden_context), dim=2)
                    logit_hidden_context_arr = []
                    if self.n_facet_emb == 0:
                        #print(projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i].size())
                        #print(projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i][0].size())
                        #print(self.lm_head.weight[input_ids_mod, :].size())
                        #print(self.lm_head.weight[input_ids_mod[0, :],:].size() )
                        #for j in range(bsz):
                        #    logit_hidden_context_arr.append(F.linear(projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i][j], self.lm_head.weight[input_ids_mod[j, :], :], None))
                        #logit_hidden_context = torch.stack(logit_hidden_context_arr, dim =0)
                        logit_hidden_context =  (projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i].unsqueeze(dim=2).expand(bsz,seq_len,seq_len+1,hidden_size) * self.lm_head.weight[input_ids_mod, :].unsqueeze(dim=1).expand(bsz,seq_len,seq_len+1,hidden_size) ).sum(dim=-1)  
                    elif self.n_facet_emb == 2:
                        #logit_hidden_context_arr = []
                        #for j in range(bsz):
                        #    logit = F.linear(projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i][j], self.lm_head.weight[input_ids[j, :], :], None)
                        #    logit += F.linear(projected_emb_arr[-2][j], emb_transform[j], None)
                        #    logit_hidden_context_arr.append(logit)
                        #logit_hidden_context = torch.stack(logit_hidden_context_arr, dim =0) #bsz, seq_len, seq_len
                        logit_hidden_context = (projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i].unsqueeze(dim=2).expand(bsz,seq_len,seq_len,hidden_size) * self.lm_head.weight[input_ids, :].unsqueeze(dim=1).expand(bsz,seq_len,seq_len,hidden_size)).sum(dim=-1)
                        logit_hidden_context += (projected_emb_arr[-2].unsqueeze(dim=2).expand(bsz,seq_len,seq_len,hidden_size) * emb_transform.unsqueeze(dim=1).expand(bsz,seq_len,seq_len,hidden_size)).sum(dim=-1)

                        temp_col = torch.full((bsz, seq_len, 1), 1000, dtype = logit_hidden_context.dtype, device=device)
                        logit_hidden_context = torch.cat((temp_col, logit_hidden_context), dim=2)
                    self.scatter_logits(logit_hidden_context, input_ids_mod, facet_lm_logits_arr, only_before_index_expand, count_duplicates_before_index, i, device)           
                
                
        #logits for n_facet (==n_facet_effective)
        # for i in range(self.n_facet_context, self.n_facet_all):
        for i in range(self.n_facet):       
            if self.starve_idx < 0 or self.starve_idx == i:
                facet_lm_logits_real_arr.append( facet_lm_logits_arr[i] )
            else:
                facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
        

        with torch.no_grad():
            if not self.only_commpute_loss:
                stacked_facet_emb = torch.stack(projected_emb_arr, dim=0)
                stacked_facet_emb = stacked_facet_emb / (1e-12 + stacked_facet_emb.norm(dim = -1, keepdim=True))
                pred_mean = stacked_facet_emb.mean(dim = 0, keepdim = True)
                div_raw = (stacked_facet_emb - pred_mean).norm(dim = -1)
                emb_div_arr = - div_raw.mean(dim=1).mean(dim=0)
                emb_div = emb_div_arr.mean()
            if vis_seq_loss or vis_simple:
                seq_len = emb_div_arr.numel()
                num_token_removed = seq_len % self.num_vis_bin
                proper_seq_len = seq_len - num_token_removed
                emb_div_arr_vis = emb_div_arr[:proper_seq_len].view(self.num_vis_bin,-1).mean(dim=-1)            
            if self.masking_ratio > 0:
                num_facet, bsz, seq_len = div_raw.size()
                var_avg_flat = div_raw.mean(0).view(-1)
                var_avg = var_avg_flat.median()
                single_facet_mask = var_avg_flat < var_avg * self.masking_ratio
        
        # #len(facet_lm_logits_arr) -> n.facet = n_facet_all - n_facet_context    
        stacked_facet_lm_logits = torch.stack(facet_lm_logits_arr, dim=0)

        #weight_mode = ''
        weight = None
        if self.weight_mode == 'dynamic':
            weight = self.weight_facet_decoder(stacked_hidden_emb_arr).softmax(dim=-1) #hidden_dim*hidden_input_state_ration -> n_facet_effective
        elif self.weight_mode == 'static':
            weight = self.weight_global.softmax(dim=-1) #torch.ones(n_facet_effective)  
        
        #cal stage1 prob
        prediction_prob = 0

        for i in range(self.n_facet_effective):
            facet_lm_logits = facet_lm_logits_real_arr[i]            
            if self.pointer_mode == "CopyNet" or self.pointer_mode == "cache":
                context_logit = torch.log( context_exp_to_add + 1e-20 )
                facet_lm_logits_context = torch.maximum(facet_lm_logits, context_logit)
                facet_lm_logits_sig = torch.exp(facet_lm_logits - facet_lm_logits_context.max(dim=-1,keepdim=True)[0]) + torch.exp( context_logit - facet_lm_logits_context.max(dim=-1,keepdim=True)[0] )
                #print("context_exp_to_add", context_exp_to_add)
                #print("log context_exp_to_add", torch.log( context_exp_to_add + 1e-10 ) )
                #print("exp log context_exp_to_add", torch.exp( torch.log( context_exp_to_add + 1e-10 ) - facet_lm_logits.max(dim=-1,keepdim=True)[0] ) )
                facet_lm_logits_softmax = facet_lm_logits_sig / facet_lm_logits_sig.sum(dim=-1,keepdim=True)
            elif self.pointer_mode == "Pointer":
                gen_weight = torch.sigmoid( self.weight_gen(hidden_emb_arr[0]) )
                facet_lm_logits_softmax = (1-gen_weight) * prob_to_add + gen_weight * facet_lm_logits.softmax(dim=-1)
            elif self.pointer_mode == "Pointer_Sentinel":
                facet_lm_logits_softmax = prob_to_add + gen_weight * facet_lm_logits.softmax(dim=-1)
            elif self.softmax_nonlinear == 'sigsoftmax': #'None' here
                facet_lm_logits_sig = torch.exp(facet_lm_logits - facet_lm_logits.max(dim=-1,keepdim=True)[0]) * (1e-20 + torch.sigmoid(facet_lm_logits))
                facet_lm_logits_softmax = facet_lm_logits_sig / facet_lm_logits_sig.sum(dim=-1,keepdim=True)
            elif self.softmax_nonlinear == 'None':
                facet_lm_logits_softmax = facet_lm_logits.softmax(dim=-1) #softmax over final logits

            if self.weight_mode == 'dynamic':
                prediction_prob += facet_lm_logits_softmax * weight[:,:,i].unsqueeze(-1)
            elif self.weight_mode == 'static':
                prediction_prob += facet_lm_logits_softmax * weight[i]
            else:
                prediction_prob += facet_lm_logits_softmax / self.n_facet_effective #softmax over final logits/1

        #for reranker stage2
        if self.n_facet_stage2>0:
            stage2_logits = self._reranker_forward(input_ids = input_ids, 
                                                   stage1_outputs = transformer_outputs, 
                                                   stage1_prob = prediction_prob.clone().detach(),
                                                   stage1_logits = facet_lm_logits_real_arr[0].clone().detach(),
                                                   labels = labels,
                                                   stage2_multiple_hidden_states = projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) +self.n_facet_context])
                
        outputs = (prediction_prob,) + (stacked_facet_lm_logits, ) + transformer_outputs[1:]
        # outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n            
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.NLLLoss(reduction='none')
            shift_prediction_prob = prediction_prob[..., :-1, :].contiguous()
            shift_labels_flat = shift_labels.view(-1)
            stage1_loss_raw = loss_fct(torch.log(shift_prediction_prob.view(-1, self.vocab_size)+1e-8), shift_labels_flat)
            stage1_loss = stage1_loss_raw[shift_labels_flat != -100].mean()
            
            if self.n_facet_stage2>0:
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_stage2_logits = stage2_logits[..., :-1, :].contiguous()
                stage2_loss_raw = loss_fct(shift_stage2_logits.view(-1, shift_stage2_logits.size(-1)), shift_labels_flat)
                stage2_loss = stage2_loss_raw[shift_labels_flat != -100].mean()
                
                loss = stage1_loss + stage2_loss
            else:
                loss = stage1_loss
                stage2_loss = stage1_loss
                
                loss_raw = stage1_loss_raw
                stage2_loss_raw = stage1_loss_raw

            dist_to_period = None
            top_val_single = None
            top_idx_single = None
            if vis_seq_loss or vis_simple:
                with torch.no_grad():
                    bsz, seq_len, vocab_size = shift_prediction_prob.size()
                    if exclude_neg_labels_for_MRR:
                        shift_labels_MRR = shift_labels.view(-1)
                        good_label_mask = shift_labels_MRR >= 0
                        shift_prediction_prob_MRR = shift_prediction_prob.view(-1,vocab_size)[good_label_mask,:]
                        gt_prob_MRR = torch.gather(shift_prediction_prob_MRR,  dim=-1 , index = shift_labels_MRR[good_label_mask].unsqueeze(dim=-1))
                        gt_rank_MRR = (gt_prob_MRR.expand_as(shift_prediction_prob_MRR) <= shift_prediction_prob_MRR).type(torch.long).sum(dim = -1)
                        seq_len_small = gt_rank_MRR.size(0)

                    else:
                        gt_prob_MRR = torch.gather(shift_prediction_prob,  dim=-1 , index = shift_labels.unsqueeze(dim=-1))
                        gt_rank_MRR = (gt_prob_MRR.expand_as(shift_prediction_prob) <= shift_prediction_prob).type(torch.long).sum(dim = -1)
                        seq_len_small = seq_len
                    num_token_removed = seq_len_small % self.num_vis_bin_loss
                    proper_seq_len = seq_len_small - num_token_removed
                    MRR_raw = (1 / gt_rank_MRR.type(torch.float))
                    if not exclude_neg_labels_for_MRR:
                        MRR_seq = MRR_raw.mean(dim=0)
                    else:
                        MRR_seq = MRR_raw
                    MRR_seq_vis = MRR_seq[:proper_seq_len].view(self.num_vis_bin_loss,-1).mean(dim=-1)
                    MRR_raw = MRR_raw.view(-1)
                    
                    num_token_removed = seq_len % self.num_vis_bin_loss
                    proper_seq_len = seq_len - num_token_removed
                    
                    if self.n_facet_stage2>0:
                        stage1_loss_seq = stage1_loss_raw.view(bsz, seq_len).mean(dim=0)
                        stage1_loss_seq_vis = stage1_loss_seq[:proper_seq_len].view(self.num_vis_bin_loss,-1).mean(dim=-1)

                        stage2_loss_seq = stage2_loss_raw.view(bsz, seq_len).mean(dim=0)
                        stage2_loss_seq_vis = stage2_loss_seq[:proper_seq_len].view(self.num_vis_bin_loss,-1).mean(dim=-1)
                    else:
                        loss_seq = loss_raw.view(bsz, seq_len).mean(dim=0)
                        loss_seq_vis = loss_seq[:proper_seq_len].view(self.num_vis_bin_loss,-1).mean(dim=-1)   
                    
                    if self.period_idx > 0:
                        dist_to_period = torch.zeros( (bsz, seq_len), device = shift_labels.device, dtype = torch.long )
                        for i in range(bsz):
                            period_position = (shift_labels[i,:] == self.period_idx).nonzero(as_tuple=True)[0]
                            num_period = period_position.numel()
                            if num_period > 0:
                                diff_pos = period_position.unsqueeze(dim=-1).expand(num_period, seq_len) - torch.arange(seq_len, device=period_position.device).unsqueeze(dim=0).expand(num_period, seq_len)
                                relative_pos = torch.abs( diff_pos )
                                #dist_to_period[i] = relative_pos.min(dim=0)[0]
                                dist_to_period[i] = torch.gather(diff_pos,0,relative_pos.min(dim=0,keepdim=True)[1])
            if vis_seq_loss:
                with torch.no_grad():
                    div_weighted_sq_raw = None
                    facet_norm = None
                    collapse_difference = None
                    collapse_difference_val = None
                    collapse_difference_inv = None
                    stacked_facet_emb_n = stacked_facet_emb

                    weighted_facet_first = weight.permute(2,0,1) # bsz, seq_len, facet -> facet, bsz, seq_len
                    stacked_facet_emb_norm = stacked_facet_emb_n / stacked_facet_emb_n.norm(dim=-1, keepdim=True)
                    pred_mean_weighted = (stacked_facet_emb_norm * weighted_facet_first.unsqueeze(dim=-1)).sum(dim=0) / weight.sum(dim=-1).unsqueeze(dim=-1)
                    div_norm_weighted = (stacked_facet_emb_norm - pred_mean_weighted).norm(dim = -1) # facet, bsz, seq_len
                    div_weighted_sq_raw = (div_norm_weighted*div_norm_weighted * weighted_facet_first).sum(dim=0) # bsz, seq_len
                    if self.n_facet > 1:
                        collapse_single_prob = self.lm_head(pred_mean_weighted).softmax(dim=-1) #bsz, seq_len, vocab
                        top_val_single, top_idx_single = torch.topk(collapse_single_prob, eval_recon_top_k, dim=-1)
                        top_val_before, top_idx_before = torch.topk(prediction_prob, eval_recon_top_k, dim=-1)
                        #top_val_org = torch.gather(prediction_prob, dim=-1 , index = top_idx)
                        top_val_now = torch.gather(prediction_prob, dim=-1 , index = top_idx_single)
                        top_val_new = torch.gather(collapse_single_prob, dim=-1 , index = top_idx_before)
                        collapse_difference_val = top_val_before.sum(dim = -1) -  top_val_now.sum(dim = -1) #torch.abs(top_val - top_val_now).sum(dim = -1)
                        collapse_difference = top_val_now.sum(dim = -1) / top_val_before.sum(dim = -1)
                        collapse_difference_inv = top_val_new.sum(dim = -1) / top_val_single.sum(dim = -1)
                    else:
                        facet_norm = projected_emb_arr[0].norm(dim=-1)

            if not self.only_commpute_loss:
                with torch.no_grad():
                    lm_logits_max, lm_max_idx = torch.max(stacked_facet_lm_logits.softmax(dim=-1), dim=0)
                    count_best_arr = torch.zeros( (1, self.n_facet_effective), device = lm_max_idx.device)
                    shift_labels = input_ids[..., 1:].contiguous()
                    best_idx = torch.gather(lm_max_idx[..., :-1, :], dim=-1 , index = shift_labels.unsqueeze(dim=-1))
                    have_label_best_idx = best_idx.squeeze(dim=-1)
                    for i in range(self.n_facet_effective):
                        count = torch.sum(have_label_best_idx == i)
                        count_best_arr[0,i] = count

            outputs = (loss,) + outputs + (stage1_loss,) + (stage2_loss,)
        
        if self.n_facet_stage2>0:
            if self.only_commpute_loss:
                return outputs
            elif vis_simple:
                return outputs, emb_div, count_best_arr, weight, emb_div_arr_vis.view(1,-1), stage1_loss_seq_vis.view(1,-1), stage2_loss_seq_vis.view(1,-1), MRR_seq_vis.view(1,-1), stage1_loss_raw, stage2_loss_raw, MRR_raw, div_raw.mean(dim=0)

            elif vis_seq_loss:
                return outputs, emb_div, count_best_arr, weight, emb_div_arr_vis.view(1,-1), stage1_loss_seq_vis.view(1,-1), stage2_loss_seq_vis.view(1,-1), MRR_seq_vis.view(1,-1), stage1_loss_raw, stage2_loss_raw, MRR_raw, div_raw.mean(dim=0), dist_to_period.view(-1), div_weighted_sq_raw, collapse_difference, collapse_difference_inv, collapse_difference_val, facet_norm, top_val_single, top_idx_single
            elif output_weight:
                return outputs, emb_div, count_best_arr, weight
            else:
                return outputs, emb_div, count_best_arr, weight
        else:
            if self.only_commpute_loss:
                return outputs
            elif vis_simple:
                return outputs, emb_div, count_best_arr, weight, emb_div_arr_vis.view(1,-1), loss_seq_vis.view(1,-1), MRR_seq_vis.view(1,-1), loss_raw, MRR_raw, div_raw.mean(dim=0)

            elif vis_seq_loss:
                return outputs, emb_div, count_best_arr, weight, emb_div_arr_vis.view(1,-1), loss_seq_vis.view(1,-1), MRR_seq_vis.view(1,-1), loss_raw, MRR_raw, div_raw.mean(dim=0), dist_to_period.view(-1), div_weighted_sq_raw, collapse_difference, collapse_difference_inv, collapse_difference_val, facet_norm, top_val_single, top_idx_single
            elif output_weight:
                return outputs, emb_div, count_best_arr, weight
            else:
                return outputs, emb_div, count_best_arr, weight

    def _reranker_forward(
        self,
        input_ids=None,
        stage1_outputs=None,
        stage1_prob=None,
        stage1_logits=None,
        labels=None,
        stage2_multiple_hidden_states=None,
    ):        
        if stage1_outputs==None:
            stage1_outputs = self.transformer(
                input_ids,
                token_type_ids = torch.zeros_like(input_ids)
            )
        hidden_states = stage1_outputs[0]
        past = stage1_outputs[1]
        
        past = torch.stack(past)
        
        bsz, seq_len, hidden_size = hidden_states.size()
        device=hidden_states.device
        
        scatter_token_type_ids_index = torch.tensor(np.arange(0, self.Block_size-1), dtype = torch.long, device=device).expand(bsz, self.Block_size, -1) #bsz, Block_size, -1
        scatter_token_type_ids_src = torch.ones(self.Block_size, self.Block_size-1, dtype = torch.long, device=device)
        scatter_token_type_ids_src = torch.triu(scatter_token_type_ids_src, diagonal=0).expand(bsz, -1, -1) #bsz, -1, -1

        scatter_input_ids_index = []
        position_ids_helper = []
        gather_output_index = []

        for i in range(self.Block_size):
            scatter_input_ids_index.append(torch.arange(i, i+self.stage2_CAN_NUM+2, dtype = torch.long, device=device))
            position_ids_helper.append(torch.cat([torch.arange(1, i+2, dtype = torch.long, device=device), torch.ones(self.stage2_CAN_NUM+self.Block_size-i, dtype = torch.long, device=device) * (i+1)], 0))
            gather_output_index.append(torch.arange(i+2, self.stage2_CAN_NUM+i+2, dtype = torch.long, device=device))

        scatter_input_ids_index = torch.stack(scatter_input_ids_index, 0).expand(bsz, -1, -1)    
        position_ids_helper = torch.stack(position_ids_helper, 0).expand(bsz, -1, -1) #bsz, -1, -1 
        gather_output_index = torch.stack(gather_output_index, 0).expand(bsz, hidden_size, -1, -1) #bsz, hidden_size, -1, -1
        gather_output_index = torch.transpose(gather_output_index, 1, 2)
        
        all_stage2_logits = []
#         all_rerank_labels = []
#         all_Block_input_ids = []
        for i in range(self.num_of_rerank):
            #get the input for this Block  
            Block_input_ids = input_ids[:, self.rerank_places[i]+1:self.rerank_places[i]+self.Block_size]
#             all_Block_input_ids.append(Block_input_ids)
            
            stage1_hidden_states = stage2_multiple_hidden_states[:, self.rerank_places[i]:self.rerank_places[i]+self.Block_size, :]
            #stage1_hidden_states = hidden_states[:, self.rerank_places[i]:self.rerank_places[i]+self.Block_size, :]
            
            if stage1_logits==None:
                stage1_logits_here = self.lm_head(stage1_hidden_states)
            else:
                stage1_logits_here = stage1_logits[:, self.rerank_places[i]:self.rerank_places[i]+self.Block_size]
            
            if stage1_prob==None:
                #get candidate token ids according to the logits
                _, candidate_token_ids = torch.topk(stage1_logits_here, self.stage2_CAN_NUM)
            else:
                stage1_prob_here = stage1_prob[:, self.rerank_places[i]:self.rerank_places[i]+self.Block_size]
                #get candidate token ids according to the prob
                _, candidate_token_ids = torch.topk(stage1_prob_here, self.stage2_CAN_NUM)
                
            rerank_labels = labels[..., self.rerank_places[i]: self.rerank_places[i]+self.Block_size]
#             all_rerank_labels.append(rerank_labels)
            
            sep_token = torch.ones(size = [bsz, self.Block_size, 1], dtype = torch.long, device=device) * (self.vocab_size-1)
            candidate_context_ids = torch.cat([sep_token, sep_token, candidate_token_ids], -1)
            rerank_input_ids = torch.ones(size = [bsz, self.Block_size, self.Block_size+self.stage2_CAN_NUM+1], dtype = torch.long, device=device) * (self.vocab_size-1)
            rerank_input_ids[:, :, :self.Block_size-1] = Block_input_ids.unsqueeze(1).expand(-1, self.Block_size, -1)
            
            rerank_input_ids = rerank_input_ids.scatter(2, scatter_input_ids_index, candidate_context_ids)
            
            #put H in stage1 before all candidates, since we will dot-product H with all candidates stage2 results
            stage1_hidden_states = self.stage1H_linear_head(stage1_hidden_states)
            rerank_input_embeds = self.wte(rerank_input_ids)
            for dim1 in range(bsz):
                for dim2 in range(self.Block_size):
                    rerank_input_embeds[dim1][dim2][dim2+1] = stage1_hidden_states[dim1][dim2]
            
            token_type_ids = torch.ones_like(rerank_input_ids)
            token_type_ids = token_type_ids.scatter(2, scatter_token_type_ids_index, scatter_token_type_ids_src)
            candidate_position_id = position_ids_helper + self.rerank_places[i]
            
            #rerank_input_ids = rerank_input_ids.reshape(bsz * Block_size, Block_size+self.stage2_CAN_NUM+1)
            rerank_input_embeds = rerank_input_embeds.reshape(bsz * self.Block_size, self.Block_size+self.stage2_CAN_NUM+1, hidden_size)
            token_type_ids = token_type_ids.reshape(bsz * self.Block_size, self.Block_size+self.stage2_CAN_NUM+1)
            candidate_position_id = candidate_position_id.reshape(bsz * self.Block_size, self.Block_size+self.stage2_CAN_NUM+1)
            
            past_here=[]
            for b in range(bsz):
                past_here_batch = past[:, :, b:b+1, :, :self.rerank_places[i]+1, :]
                past_here_batch = past_here_batch.expand(-1, -1, self.Block_size, -1, -1, -1)
                past_here.append(past_here_batch)
            past_here = torch.cat(past_here, dim=2)
            
            #get output from gpt2            
            rerank_outputs = self.transformer(inputs_embeds=rerank_input_embeds,
                                              past=past_here,
                                              token_type_ids=token_type_ids,
                                              position_ids = candidate_position_id
                                  )
            
            rerank_hidden_states = rerank_outputs[0].reshape(bsz, self.Block_size, self.Block_size+self.stage2_CAN_NUM+1, hidden_size)
            rerank_hidden_states = self.rerank_linear_head(rerank_hidden_states)
            rerank_hidden_states = torch.transpose(rerank_hidden_states, 2, 3)
            rerank_hidden_states = rerank_hidden_states.gather(3, gather_output_index)
            rerank_hidden_states = torch.transpose(rerank_hidden_states, 2, 3)
            
            stage2_logits = stage1_logits_here.clone().detach()
            stage2_candidates_logits = torch.matmul(rerank_hidden_states, stage1_hidden_states.unsqueeze(-1)).squeeze(-1)            
            
            if self.reranker_stage2_efficient_mode == 'assign_reranker_stage2_logits':
                stage2_logits = stage2_logits.scatter(2, candidate_token_ids, stage2_candidates_logits)
            else:
                stage2_logits = stage2_logits.scatter_add(2, candidate_token_ids, stage2_candidates_logits)

            all_stage2_logits.append(stage2_logits)
        
        all_stage2_logits = torch.cat(all_stage2_logits, 1)
        
        return all_stage2_logits




