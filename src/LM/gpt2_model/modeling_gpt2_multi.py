# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""


import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
import random

from .configuration_gpt2 import GPT2Config
from .file_utils import add_start_docstrings
from .modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer


logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
    "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-pytorch_model.bin",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin",
}


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        output_attn = self.attn(
            self.ln_1(x), layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    config_class = GPT2Config
    pretrained_model_archive_map = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


GPT2_START_DOCSTRING = r"""    OpenAI GPT-2 model was proposed in
    `Language Models are Unsupervised Multitask Learners`_
    by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
    It's a causal (unidirectional) transformer pre-trained using  language modeling on a very large
    corpus of ~40 GB of text data.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Language Models are Unsupervised Multitask Learners`:
        https://openai.com/blog/better-language-models/

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            GPT-2 is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.GPT2Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
    GPT2_INPUTS_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(2, batch_size, num_heads, sequence_length, embed_size_per_head)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):
        super(GPT2Model, self).__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.output_past = config.output_past

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
            )

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


class GPT2MoSLMHeadModel(GPT2PreTrainedModel):
    #def __init__(self, config, transformer_input, n_facet, n_facet_hidden, weight_mode, use_proj_bias=False, n_facet_window=0, n_facet_MLP=0, vis_seq_loss=False, softmax_nonlinear='sigsoftmax'):
    def __init__(self, config, transformer_input, n_facet, n_facet_hidden, weight_mode, use_proj_bias=False, n_facet_window=0, n_facet_MLP=0, only_commpute_loss=False, softmax_nonlinear='None', efficient_mode='None', masking_ratio=-1, seq_len = 0, device=None, n_facet_effective_in=-1, last_num=0, share_facet_linear=False, individual_hidden_states=False):
        super(GPT2MoSLMHeadModel, self).__init__(config)
        #self.transformer = GPT2Model(config)
        self.n_facet = n_facet
        self.n_facet_hidden = n_facet_hidden
        assert n_facet_MLP <= 0
        assert n_facet_window <= 0
        n_facet_window = - n_facet_window
        n_facet_MLP = - n_facet_MLP
        self.n_facet_MLP = n_facet_MLP
        self.n_facet_window = n_facet_window
        self.softmax_nonlinear=softmax_nonlinear
        self.efficient_mode = efficient_mode
        self.masking_ratio = masking_ratio
        self.individual_hidden_states = individual_hidden_states
        if individual_hidden_states:
            assert n_facet == n_facet_hidden
            assert n_facet_window == 0
            assert n_facet_MLP == 0
            assert efficient_mode=='None'
        if n_facet_hidden == 0:
            n_facet_hidden = n_facet
        #self.project_arr = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=False) for i in range(n_facet)])
        if n_facet_MLP > 0:
            hidden_state_input_ratio = 1 + n_facet_MLP
            #self.MLP_linear = nn.Linear(config.n_embd * (n_facet_hidden + n_facet_window), config.n_embd * n_facet_MLP)
            self.MLP_linear = nn.Linear(config.n_embd * (n_facet_hidden * (n_facet_window+1) ), config.n_embd * n_facet_MLP)
            self.MLP_linear_l2 = nn.Linear(config.n_embd * n_facet_MLP, config.n_embd * n_facet_MLP)
        else:
            #hidden_state_input_ratio = n_facet_hidden+n_facet_window
            hidden_state_input_ratio = n_facet_hidden * (n_facet_window+1)
        if individual_hidden_states:
            total_lin_dim = config.n_embd
        else:
            total_lin_dim = config.n_embd * hidden_state_input_ratio
        small_value = 0.0001
        #small_value = 1e-10
        #small_value = 0.01
        self.share_facet_linear = share_facet_linear
        if share_facet_linear:
            self.project_arr = nn.ModuleList([nn.Linear(total_lin_dim - config.n_embd, config.n_embd, bias=use_proj_bias) for i in range(n_facet)])
            self.shared_projection = nn.Linear(n_facet_hidden, config.n_embd, bias=use_proj_bias)
            self.shared_projection.weight.data = torch.eye(config.n_embd)
            if use_proj_bias:
                self.shared_projection.bias.data.zero_()
            for i in range(n_facet):
                if use_proj_bias:
                    self.project_arr[i].bias.data.zero_()
                linear_weights = torch.zeros_like(self.project_arr[i].weight.data)
                if i!= n_facet - 1:
                    linear_weights = linear_weights + small_value * (torch.rand((config.n_embd, total_lin_dim  - config.n_embd)) - 0.5 )
                self.project_arr[i].weight.data = linear_weights
        else:
            self.project_arr = nn.ModuleList([nn.Linear(total_lin_dim, config.n_embd, bias=use_proj_bias) for i in range(n_facet)])
            #self.project_arr[0].weight.data = torch.eye(config.n_embd)
            #self.project_arr[0].weight.data = torch.eye(config.n_embd)
            #for i in range(1,n_facet):
            #num_init_facet = n_facet
            #if n_facet == 3:
            #    num_init_facet = 2
            for i in range(n_facet):
                if use_proj_bias:
                    self.project_arr[i].bias.data.zero_()
                linear_weights = torch.zeros_like(self.project_arr[i].weight.data)
                #if i!= n_facet - 1 or i!= n_facet - 2:
                if i!= n_facet - 1:
                #if True:
                    linear_weights = linear_weights + small_value * (torch.rand((config.n_embd, total_lin_dim)) - 0.5 )
                linear_weights[:,:config.n_embd] = torch.eye(config.n_embd)
                self.project_arr[i].weight.data = linear_weights
                #self.project_arr[i].weight.data = torch.eye(config.n_embd) + small_value * (torch.rand((config.n_embd, config.n_embd)) - 0.5 )
                #if i <num_init_facet:
                #    self.project_arr[i].weight.data = torch.eye(config.n_embd) + small_value * (torch.rand((config.n_embd, config.n_embd)) - 0.5 )
                #else:
                #    self.project_arr[i].weight.data = 10 * (torch.rand((config.n_embd, config.n_embd)) - 0.5 )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        n_facet_effective = n_facet
        if efficient_mode == 'context' or efficient_mode == 'context_merge':
            #device = self.project_arr[0].weight.data.device
            #print(device)
            only_before_index = torch.tensor( list(range(seq_len)), device=device )
            only_before_index = torch.tril(only_before_index.expand(seq_len,seq_len))
            self.only_before_index = only_before_index
            if efficient_mode == 'context':
                self.other_word_logit = nn.Parameter( torch.ones(1, dtype=torch.float) )
            elif efficient_mode == 'context_merge':
                n_facet_effective = n_facet - 1
                self.weight_context = nn.Linear(config.hidden_size * hidden_state_input_ratio, 1)
                self.weight_context.weight.data *= 0.0001 
                self.weight_context.bias.data[:] = 0
        elif efficient_mode == 'even_multi_alter':
            self.last_num = last_num

            self.n_facet_effective = n_facet_effective_in
            n_facet_effective = n_facet_effective_in
        elif efficient_mode == 'even_multi_shuffle':
            self.last_num = last_num

            self.n_facet_effective = n_facet_effective_in
            n_facet_effective = n_facet_effective_in
            #self.shuffle_block_idx_arr = []
            #for i in range(self.n_facet_effective):
            #    shuffle_block_idx = list(range(self.n_facet))
            #    random.shuffle(shuffle_block_idx)
            #    self.shuffle_block_idx_arr.append(shuffle_block_idx)
            #print(self.shuffle_block_idx_arr)
            
            #self.shuffle_vocab_idx_arr = []
            #self.shuffle_rev_idx_arr = []
            #for i in range(self.n_facet_effective):
            #    shuffle_vocab_idx = list(range(config.vocab_size))
            #    random.shuffle(shuffle_vocab_idx)
            #    shuffle_vocab_idx_tensor = torch.tensor(shuffle_vocab_idx,device=device)
            #    self.shuffle_vocab_idx_arr.append(shuffle_vocab_idx_tensor)
            #    shuffle_rev_idx = torch.empty_like(shuffle_vocab_idx_tensor)
            #    shuffle_rev_idx[shuffle_vocab_idx_tensor] = torch.arange(config.vocab_size,device=device)
            #    self.shuffle_rev_idx_arr.append(shuffle_rev_idx)
                
        elif efficient_mode == 'even_last_2':
            n_facet_effective = n_facet_effective_in
        elif efficient_mode == 'even' or efficient_mode == 'even_last':
            n_facet_effective = n_facet_effective_in

        if len(weight_mode) > 0:
            self.weight_facet_decoder = nn.Linear(config.hidden_size * hidden_state_input_ratio, n_facet_effective)
            #self.weight_facet_decoder = nn.Linear(config.hidden_size * n_facet_hidden * (n_facet_window+1), n_facet)
            self.weight_global = nn.Parameter( torch.ones(n_facet_effective) )

        #self.init_weights()
        self.weight_mode = weight_mode
        self.transformer = transformer_input
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.output_probs = True
         
        #self.vis_seq_loss = vis_seq_loss
        #self.vis_simple = vis_simple
        self.only_commpute_loss = only_commpute_loss
        self.starve_idx = -1
        self.num_vis_bin = 4
        self.num_vis_bin_loss = 5
        self.period_idx = -1
    
    def get_facet_emb(self,input_emb, i):
        if self.individual_hidden_states:
            rev_i = len(self.project_arr) - i
            return self.project_arr[i](input_emb[:,:,(rev_i-1)*self.n_embd: rev_i*self.n_embd])
            #return self.project_arr[i](input_emb[rev_i-1])
        elif self.share_facet_linear:
            return self.project_arr[i](input_emb[:,:,self.n_embd:]) + self.shared_projection(input_emb[:,:,:self.n_embd])
        else:
            return self.project_arr[i](input_emb)

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_weight=True,
        eval_recon_top_k=None, 
        vis_simple=False,
        vis_seq_loss = False,
        exclude_neg_labels_for_MRR=False
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        #hidden_states = transformer_outputs[0]
        all_hidden_states = transformer_outputs[2]
 
        hidden_emb_arr = []
        #for i in range(self.n_facet_hidden):
        #    hidden_states = all_hidden_states[-(i+1)]
        #    hidden_emb_arr.append(hidden_states)
        #
        #for i in range(self.n_facet_window):
        #    hidden_states = all_hidden_states[-1]
        #    bsz, seq_len, hidden_size = hidden_states.size()
        #    device = hidden_states.device
        #    shifted_hidden = torch.cat( (torch.zeros( (bsz, (i+1), hidden_size), device = device), hidden_states[:,:-(i+1),:]), dim = 1)
        #    hidden_emb_arr.append(shifted_hidden)
        
        ## Multi-input hidden states: generate q_ct from hidden states
        # h_facet_hidden -> H = 3, n_face_window -> W = 2
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
            if self.individual_hidden_states:
                stacked_hidden_emb_arr = torch.cat(hidden_emb_arr, dim=-1)
            else:
                stacked_hidden_emb_arr = hidden_emb_arr[0]

        
        facet_lm_logits_real_arr = []
        facet_lm_logits_arr = []
        #projected_emb_real_arr = []
        projected_emb_arr = []
        n_facet_effective = self.n_facet

        if self.masking_ratio > 0:
            for i in range(self.n_facet):
                #hidden_states = all_hidden_states[-1]
                #projected_emb = self.project_arr[i](stacked_hidden_emb_arr)
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, i)
                projected_emb_arr.append(projected_emb)
        elif self.efficient_mode == 'context_merge':
            # set to 3 in "even_last_2", 5 here
            n_facet_effective = self.n_facet - 1
            assert self.lm_head.bias is None
            bsz, seq_len, hidden_size = all_hidden_states[-1].size() #topmost hidden state embedding
            #projected_emb =  self.weight_context(stacked_hidden_emb_arr) * self.project_arr[0](stacked_hidden_emb_arr)
            # #self.weight_context = nn.Linear(config.hidden_size * hidden_state_input_ratio, 1), weights*small number, bias = 0, 
            # #hidden_state_input_ratio = 1+n_facet_MLP if n_facet_MLP>0 or H*(W+1)
            projected_emb =  self.weight_context(stacked_hidden_emb_arr) * self.get_facet_emb(stacked_hidden_emb_arr, 0) #L^f linear transformation similar to paper
            projected_emb_arr.append(projected_emb) 
            for i in range(1, self.n_facet):
                #projected_emb = self.project_arr[i](stacked_hidden_emb_arr)
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, i)
                projected_emb_arr.append(projected_emb)
                facet_lm_logits_arr.append( self.lm_head( projected_emb ) )
            # #projected_emb_arr -> (n_facet, bsz, seq_len, hidden_size)
            # #facet_lm_logits_arr -> (n_facet-1, bsz, seq_len, vocab_size)

            for j in range(bsz):
                logit_hidden_context = F.linear( projected_emb_arr[0][j,:,:], self.lm_head.weight[input_ids[j,:],:], None) #lm_head.weight -> (vocab_size, hidden_size), input_id -> (bsz, seq_len)
                # #logit_hidden_context -> (seq_len, hidden_size)*(hidden_size, seq_len) last dim is seq_len instead of vocab_size -> first facet is used to compute only the probabilities of the words in the context
                # #only_before_index -> (seq_len, seq_len), loweer triangle, diagonal [0 -> seq_len-1]
                # #gather: logit_hidden_context_before[i][j] = logit_hidden_context[i][ index[i][j][k] ]
                logit_hidden_context_before = torch.gather(logit_hidden_context, dim=1, index=self.only_before_index) # (seq_len, seq_len)
                temp = facet_lm_logits_arr[0][j, :, input_ids[j,0]] #(seq_len, 1)
                #facet_lm_logits_arr[0][j, :, :].scatter_(dim=1, index= input_ids[j,self.only_before_index],src=logit_hidden_context_before)
                # #the context-dependent logits from the first facet are added to the logits of the second facet
                print("batch ", j, "scatter_add_ dim check", facet_lm_logits_arr[0][j, :, :].size(), input_ids[j,self.only_before_index].size(), logit_hidden_context_before.size())
                # #scatter_sdd_ self[i][index[i][j][k]][k] += src[i][j][k]
                facet_lm_logits_arr[0][j, :, :].scatter_add_(dim=1, index= input_ids[j,self.only_before_index],src=logit_hidden_context_before)
                
                #for k in range(seq_len):
                #    #for q in range(k+1):
                #    q_arr = list(range(k+1))
                #    logit_full[k,input_ids[j,q_arr]] = logit_hidden_context[k,q_arr]
                
                facet_lm_logits_arr[0][j, :, input_ids[j,0]] = temp + logit_hidden_context[:,0] #scatter_add would add to 0, 
            for i in range( self.n_facet - 1):

                if self.starve_idx < 0 or self.starve_idx == i:
                    facet_lm_logits_real_arr.append( facet_lm_logits_arr[i] )
                else:
                    facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
                    

        elif self.efficient_mode == 'context':
            assert self.lm_head.bias is None
            bsz, seq_len, hidden_size = all_hidden_states[-1].size()
            #projected_emb = self.project_arr[0](stacked_hidden_emb_arr)
            projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, 0) #bsz, seq_len, hidden_size
            projected_emb_arr.append(projected_emb)
            logit_arr = []
            for j in range(bsz):
                logit_hidden_context = F.linear(projected_emb[j,:,:], self.lm_head.weight[input_ids[j,:],:], None) # (seq_len, seq_len)
                logit_full = torch.zeros( (seq_len, self.vocab_size), device=all_hidden_states[-1].device ) 
                logit_full[:] = self.other_word_logit[0] #self.other_word_logit[0]=1, logit_full -> tensor of ones
                #logit_full[:] = -1e10
                #for k in range(seq_len):
                #    #for q in range(k+1):
                #    q_arr = list(range(k+1))
                #    logit_full[k,input_ids[j,q_arr]] = logit_hidden_context[k,q_arr]
                logit_hidden_context_before = torch.gather(logit_hidden_context, dim=1, index=self.only_before_index) #(seq_len, seq_len), lower triangular matrix of logit_hidden_context, rest are filled by logit_hidden_context[i][0]
                logit_full.scatter_(dim=1, index= input_ids[j,self.only_before_index],src=logit_hidden_context_before) #assign logits of context words
                
                logit_full[:,input_ids[j,0]] = logit_hidden_context[:,0] #why?
                #logit_full[:,input_ids[j,0]] = -1e10
                logit_arr.append(logit_full)
            logit_full_all = torch.stack(logit_arr,dim=0) #bsz, seq_len, vocab_size

            facet_lm_logits_arr.append(logit_full_all) #first head with ones, and logits only for prev words in context
            if self.starve_idx < 0 or self.starve_idx == 0:
                facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
            else:
                facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
                
            for i in range(1, self.n_facet):
                #projected_emb = self.project_arr[i](stacked_hidden_emb_arr)
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, i) #rest of the facets
                projected_emb_arr.append(projected_emb)

                facet_lm_logits_arr.append( self.lm_head( projected_emb ) ) #rest of n_facet-1 heads -> logits for all words in vocab
                if self.starve_idx < 0 or self.starve_idx == i:
                    facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
                else:
                    facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
        elif self.efficient_mode == 'even_multi_alter':
            n_facet_effective = self.n_facet_effective
            n_facet_not_last = self.n_facet - self.last_num
            n_facet_partition = self.n_facet_effective - self.last_num 
            assert n_facet_not_last % n_facet_partition == 0
            assert self.lm_head.bias is None
            n_facet_each = int(n_facet_not_last / n_facet_partition)
            bsz, seq_len, hidden_size = all_hidden_states[-1].size()
            block_size = int(math.ceil(self.vocab_size / n_facet_each))
            subblock_size = int(block_size / (n_facet_partition - 1) )
            for k in range(n_facet_partition):
                logit_all = torch.empty( (bsz, seq_len, self.vocab_size) , device=all_hidden_states[-1].device )

                for i in range(n_facet_each):
                    #projected_emb = self.project_arr[k*n_facet_each+i](stacked_hidden_emb_arr)
                    projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, k*n_facet_each+i)
                    projected_emb_arr.append(projected_emb)
                    #logit_all[:,:,i*block_size:(i+1)*block_size] = F.linear(projected_emb, self.lm_head.weight[i*block_size:(i+1)*block_size,:], None)
                    if k == n_facet_partition - 1:
                        logit_all[:,:,i::n_facet_each] = F.linear(projected_emb, self.lm_head.weight[i::n_facet_each,:], None)
                    else:
                        if k*subblock_size+(i+1)*block_size < self.vocab_size:
                            logit_all[:,:,k*subblock_size+i*block_size:k*subblock_size+(i+1)*block_size] = F.linear(projected_emb, self.lm_head.weight[k*subblock_size+i*block_size:k*subblock_size+(i+1)*block_size,:], None)
                        else:
                            logit_all[:,:,k*subblock_size+i*block_size:] = F.linear(projected_emb, self.lm_head.weight[k*subblock_size+i*block_size:,:], None)
                            logit_all[:,:,:k*subblock_size+(i+1)*block_size-self.vocab_size+1] = F.linear(projected_emb, self.lm_head.weight[:k*subblock_size+(i+1)*block_size-self.vocab_size+1,:], None)
                            
                facet_lm_logits_arr.append(logit_all)
                if self.starve_idx < 0 or self.starve_idx == k:
                    facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
                else:
                    facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
            for i in range(self.last_num):
                #projected_emb = self.project_arr[n_facet_not_last+i](stacked_hidden_emb_arr)
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,n_facet_not_last+i)
                projected_emb_arr.append(projected_emb)

                facet_lm_logits_arr.append( self.lm_head( projected_emb ) )
                if self.starve_idx < 0 or self.starve_idx == i+1:
                    facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
                else:
                    facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))

                    
        elif self.efficient_mode == 'even_multi_shuffle':
            #n_facet_effective = 1
            #n_facet_effective = 2
            n_facet_effective = self.n_facet_effective
            n_facet_not_last = self.n_facet - self.last_num
            n_facet_partition = self.n_facet_effective - self.last_num
            assert n_facet_not_last % n_facet_partition == 0
            assert self.lm_head.bias is None
            n_facet_each = int(n_facet_not_last / n_facet_partition)
            bsz, seq_len, hidden_size = all_hidden_states[-1].size()
            #temp_arr = [ [[] for i in range(n_facet_each) ] for k in range(n_facet_partition) ]
            #for k in range(n_facet_partition):
            #    for i in range(n_facet_each):
            #        projected_emb = self.project_arr[k*n_facet_each+i](stacked_hidden_emb_arr)
            #        projected_emb_arr.append(projected_emb)
            #        for j in range(n_facet_partition):
            #            start_idx = self.shuffle_block_idx_arr[k][i*n_facet_partition+j]
            #            temp_arr[k][i].append(F.linear(projected_emb, self.lm_head.weight[start_idx::self.n_facet,:], None))
            #for k in range(n_facet_partition):
            #    logit_all = torch.empty( (bsz, seq_len, self.vocab_size) , device=all_hidden_states[-1].device )
            #    for i in range(n_facet_each):
            #        for j in range(n_facet_partition):
            #            start_idx = self.shuffle_block_idx_arr[k][i*n_facet_partition+j]
            #            logit_all[:,:,start_idx::self.n_facet] = temp_arr[k][i][j]
            #    facet_lm_logits_arr.append(logit_all)
            #    if self.starve_idx < 0 or self.starve_idx == k:
            #        facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
            #    else:
            #        facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
            block_size = int(math.ceil(self.vocab_size / n_facet_each))
            subblock_size = int(block_size / n_facet_partition)
            for k in range(n_facet_partition):
                logit_all = torch.empty( (bsz, seq_len, self.vocab_size) , device=all_hidden_states[-1].device )
                #shuffled_lm_weight = self.lm_head.weight[self.shuffle_vocab_idx_arr[k],:]

                for i in range(n_facet_each):
                    #projected_emb = self.project_arr[k*n_facet_each+i](stacked_hidden_emb_arr)
                    projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,k*n_facet_each+i)

                    projected_emb_arr.append(projected_emb)
                    #logit_all[:,:,i::n_facet_each] = F.linear(projected_emb, self.lm_head.weight[i::n_facet_each,:], None)
                    #logit_all[:,:,i*block_size:(i+1)*block_size] = F.linear(projected_emb, self.lm_head.weight[i*block_size:(i+1)*block_size,:], None)
                    if k*subblock_size+(i+1)*block_size < self.vocab_size:
                        logit_all[:,:,k*subblock_size+i*block_size:k*subblock_size+(i+1)*block_size] = F.linear(projected_emb, self.lm_head.weight[k*subblock_size+i*block_size:k*subblock_size+(i+1)*block_size,:], None)
                    else:
                        logit_all[:,:,k*subblock_size+i*block_size:] = F.linear(projected_emb, self.lm_head.weight[k*subblock_size+i*block_size:,:], None)
                        logit_all[:,:,:k*subblock_size+(i+1)*block_size-self.vocab_size+1] = F.linear(projected_emb, self.lm_head.weight[:k*subblock_size+(i+1)*block_size-self.vocab_size+1,:], None)
                        
                    #for j in range(n_facet_partition):
                    #    start_idx = self.shuffle_block_idx_arr[k][i*n_facet_partition+j]
                    #    logit_all[:,:,start_idx::self.n_facet] = F.linear(projected_emb, self.lm_head.weight[start_idx::self.n_facet,:], None)
                    #logit_all[:,:,i::n_facet_each] = F.linear(projected_emb, shuffled_lm_weight[i::n_facet_each,:], None)
                #facet_lm_logits_arr.append(logit_all[:,:,self.shuffle_rev_idx_arr[k]])
                facet_lm_logits_arr.append(logit_all)
                if self.starve_idx < 0 or self.starve_idx == k:
                    facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
                else:
                    facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
            for i in range(self.last_num):
                #projected_emb = self.project_arr[-(i+1)](stacked_hidden_emb_arr)
                #projected_emb = self.project_arr[n_facet_not_last+i](stacked_hidden_emb_arr)
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, n_facet_not_last+i)
                projected_emb_arr.append(projected_emb)
                #projected_emb_real_arr.append(projected_emb)

                facet_lm_logits_arr.append( self.lm_head( projected_emb ) )
                if self.starve_idx < 0 or self.starve_idx == i+1:
                    facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
                else:
                    facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))

                    
            
        elif self.efficient_mode == 'even':
            n_facet_effective = 1
            #n_facet_effective = 2
            bsz, seq_len, hidden_size = all_hidden_states[-1].size()
            logit_all = torch.empty( (bsz, seq_len, self.vocab_size) , device=all_hidden_states[-1].device )
            #avg_emb = torch.zeros( (bsz, seq_len, hidden_size) , device=all_hidden_states[-1].device )
            for i in range(self.n_facet):
                #projected_emb = self.project_arr[i](stacked_hidden_emb_arr)
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, i)
                #avg_emb = avg_emb + projected_emb
                projected_emb_arr.append(projected_emb)
                #projected_emb_real_arr.append(projected_emb)
                if self.lm_head.bias is None:
                    logit_all[:,:,i::self.n_facet] = F.linear(projected_emb, self.lm_head.weight[i::self.n_facet,:], None)
                else:
                    logit_all[:,:,i::self.n_facet] = F.linear(projected_emb, self.lm_head.weight[i::self.n_facet,:], self.lm_head.bias[i::self.n_facet])
            facet_lm_logits_arr.append(logit_all)
            facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
            
            #avg_emb = avg_emb / self.n_facet
            #facet_lm_logits_arr.append( self.lm_head(avg_emb) )
            #facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )

        ## multi-softmax + multi-partition
        elif self.efficient_mode == 'even_last_2':
            #n_facet_effective = 1
            n_facet_effective = 3
            #n_facet_effective = 2
            bsz, seq_len, hidden_size = all_hidden_states[-1].size()
            logit_all = torch.empty( (bsz, seq_len, self.vocab_size) , device=all_hidden_states[-1].device )
            n_facet_not_last = self.n_facet - (n_facet_effective-1) # 6 - (3-1) = 4 -> partitions
            for i in range(n_facet_not_last):
                #projected_emb = self.project_arr[i](stacked_hidden_emb_arr)
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,i) #bsz, seq_len, n_embd
                # stacked_hidden_emb_arr -> sz, seq_len, 2*hidden_size 
                # same as project_arr? output_dim -> n_embd, 6 linear models, weights are zero for last one
                projected_emb_arr.append(projected_emb) #4 partitions
                #projected_emb_real_arr.append(projected_emb)
                #self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
                if self.lm_head.bias is None:
                    logit_all[:,:,i::n_facet_not_last] = F.linear(projected_emb, self.lm_head.weight[i::n_facet_not_last,:], None)
                else:
                    logit_all[:,:,i::n_facet_not_last] = F.linear(projected_emb, self.lm_head.weight[i::n_facet_not_last,:], self.lm_head.bias[i::n_facet_not_last])
            facet_lm_logits_arr.append(logit_all)
            if self.starve_idx < 0 or self.starve_idx == 0:
                facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
            else:
                facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
            #last two softmax, project_arr -> L^f
            for i in range(n_facet_effective-1):
                projected_emb = self.project_arr[-(i+1)](stacked_hidden_emb_arr)
                #projected_emb = self.project_arr[n_facet_not_last+i](stacked_hidden_emb_arr)
                #projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,n_facet_not_last+i)
                projected_emb_arr.append(projected_emb) 
                #projected_emb_real_arr.append(projected_emb)

                facet_lm_logits_arr.append( self.lm_head( projected_emb ) )
                if self.starve_idx < 0 or self.starve_idx == i+1:
                    facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
                else:
                    facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
        #project_emb_arr -> n_facet, bsz, seq_len, n_embd
        #facet_lm_logits_arr -> n_facet_effective, bsz, seq_len, vocab_size
        elif self.efficient_mode == 'even_last':
            #n_facet_effective = 1
            n_facet_effective = 2
            bsz, seq_len, hidden_size = all_hidden_states[-1].size()
            logit_all = torch.empty( (bsz, seq_len, self.vocab_size) , device=all_hidden_states[-1].device )
            n_facet_not_last = self.n_facet-1
            for i in range(n_facet_not_last):
                #projected_emb = self.project_arr[i](stacked_hidden_emb_arr)
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,i)
                projected_emb_arr.append(projected_emb)
                #projected_emb_real_arr.append(projected_emb)
                if self.lm_head.bias is None:
                    logit_all[:,:,i::n_facet_not_last] = F.linear(projected_emb, self.lm_head.weight[i::n_facet_not_last,:], None)
                else:
                    logit_all[:,:,i::n_facet_not_last] = F.linear(projected_emb, self.lm_head.weight[i::n_facet_not_last,:], self.lm_head.bias[i::n_facet_not_last])
            facet_lm_logits_arr.append(logit_all)
            if self.starve_idx < 0 or self.starve_idx == 0:
                facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
            else:
                facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
            
            #projected_emb = self.project_arr[-1](stacked_hidden_emb_arr)
            projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,-1)
            projected_emb_arr.append(projected_emb)
            #projected_emb_real_arr.append(projected_emb)

            facet_lm_logits_arr.append( self.lm_head( projected_emb ) )
            if self.starve_idx < 0 or self.starve_idx == 1:
                facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
            else:
                facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))

        else:
            for i in range(self.n_facet):
                #hidden_states = all_hidden_states[-1]
                #projected_emb = self.project_arr[i](stacked_hidden_emb_arr)
                projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,i)
                projected_emb_arr.append(projected_emb)
                facet_lm_logits_arr.append( self.lm_head(projected_emb) )
                if self.starve_idx < 0 or self.starve_idx == i:
                    facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
                    #projected_emb_real_arr.append(projected_emb)
                else:
                    bsz, seq_len, hidden_size = all_hidden_states[-1].size()
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
        
        if self.masking_ratio > 0:
            avg_emb = torch.mean( torch.stack(projected_emb_arr), dim=0)
            avg_emb_flat = avg_emb.view(bsz * seq_len, self.n_embd)
            single_facet_logit = self.lm_head( avg_emb_flat[single_facet_mask,:] )
            #if self.starve_idx >= 0:
            #    projected_emb = projected_emb_arr[self.starve_idx]
            #    logit_starve = self.lm_head(projected_emb)
            #    for i in range(self.n_facet):
            #        if i == self.starve_idx:
            #            facet_lm_logits_arr.append(logit_starve)
            #        else:
            #            facet_lm_logits_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
            #        facet_lm_logits_real_arr.append(logit_starve)
            #        #facet_lm_logits_real_arr.append(facet_lm_logits_arr[-1])
            #else:
            for i in range(self.n_facet):
                projected_emb = projected_emb_arr[i]
                logit_i = torch.empty( (bsz * seq_len, self.vocab_size) , device=all_hidden_states[-1].device )
                logit_i[single_facet_mask,:] = single_facet_logit
                logit_i[~single_facet_mask,:] = self.lm_head( projected_emb.view(bsz * seq_len, self.n_embd)[~single_facet_mask,:] )

                facet_lm_logits_arr.append( logit_i.view(bsz , seq_len, self.vocab_size) )
                #facet_lm_logits_real_arr.append(facet_lm_logits_arr[-1])
                if self.starve_idx < 0 or self.starve_idx == i:
                    facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
                else:
                    facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
            #for i in range(self.n_facet):
            #    if self.starve_idx < 0 or self.starve_idx == i:
            #        facet_lm_logits_real_arr.append( facet_lm_logits_arr[i] )
            #    else:
            #        facet_lm_logits_real_arr.append( facet_lm_logits_arr[self.starve_idx] )
                    
        
        self.starve_idx = -1
        stacked_facet_lm_logits = torch.stack(facet_lm_logits_arr, dim=0)
        #stacked_facet_lm_real_logits = torch.stack(facet_lm_logits_real_arr, dim=0)
        #lm_logits_max, lm_max_idx = torch.max(stacked_facet_lm_logits, dim=0)
        #lm_logits = torch.logsumexp(stacked_facet_lm_real_logits, dim=0)
        
        
        #stacked_facet_real_emb = torch.stack(projected_emb_real_arr, dim=0)
        #stacked_facet_emb_real_mean = stacked_facet_real_emb.mean(dim=0)
        #lm_logits = (lm_logits + self.lm_head(stacked_facet_emb_real_mean) ) /2
        #with torch.no_grad():
        #    top_k = 100
        #    #count_arr = []
        #    #count_arr = np.zeros(self.n_facet)
        #    count_arr = torch.zeros( (1,n_facet_effective), device = lm_max_idx.device)
        #    for i in range(n_facet_effective):
        #        #count = torch.sum(lm_max_idx[:,:,:top_k] == i).item()
        #        count = torch.sum(lm_max_idx[:,:,:top_k] == i)
        #        count_arr[0,i] = count
        #        #count_arr.append(count)
        #

        #self.multi_project(hidden_states)
        #lm_logits = self.lm_head(hidden_states)
            
        if self.weight_mode == 'dynamic':
            #weight = F.relu(self.weight_facet_decoder(outputs[2][-1])) + 0.001
            #weight = weight / weight.sum(dim=-1, keepdim=True)
            #weight = self.weight_facet_decoder(all_hidden_states[-1]).softmax(dim=-1)
            weight = self.weight_facet_decoder(stacked_hidden_emb_arr).softmax(dim=-1)
            #weight = self.weight_facet_decoder(stacked_hidden_emb_raw_arr).softmax(dim=-1)
            #print(weight.size())
            #weight = weight[:,:-1,:].contiguous().view(-1,self.n_facet)
        elif self.weight_mode == 'static':
            #weight = F.relu(self.weight_global) + 0.001
            #weight = weight / weight.sum(dim=-1, keepdim=True)
            weight = self.weight_global.softmax(dim=-1)
        
        prediction_prob = 0
        #print(facet_lm_logits_arr[i].size())
        for i in range(n_facet_effective):
            facet_lm_logits = facet_lm_logits_real_arr[i]
            if self.softmax_nonlinear == 'sigsoftmax':
                facet_lm_logits_sig = torch.exp(facet_lm_logits - facet_lm_logits.max(dim=-1,keepdim=True)[0]) * (1e-20 + torch.sigmoid(facet_lm_logits))
                #facet_lm_logits_sig = torch.exp(facet_lm_logits) * (1e-20 + torch.sigmoid(facet_lm_logits))
                #facet_lm_logits_sig = torch.exp(facet_lm_logits - facet_lm_logits.max(dim=-1,keepdim=True)[0]) * torch.sigmoid(facet_lm_logits)
                #facet_lm_logits_sig[facet_lm_logits_sig == 0] = 1e-20
                #facet_lm_logits_sig_sum = facet_lm_logits_sig.sum(dim=-1)
                #corner_case = facet_lm_logits_sig_sum==0
                #facet_lm_logits_sig[ corner_case.unsqueeze(dim=-1).expand_as(facet_lm_logits_sig) ] = 1
                #facet_lm_logits_sig_sum[corner_case] = n_facet_effective
                #facet_lm_logits_softmax = facet_lm_logits_sig / facet_lm_logits_sig_sum.unsqueeze(dim=-1)
                facet_lm_logits_softmax = facet_lm_logits_sig / facet_lm_logits_sig.sum(dim=-1,keepdim=True)
            elif self.softmax_nonlinear == 'None':
                facet_lm_logits_softmax = facet_lm_logits.softmax(dim=-1)
            if self.weight_mode == 'dynamic':
                #prediction_prob += facet_lm_logits_real_arr[i][..., :-1, :].contiguous().view(-1, self.vocab_size).softmax(dim=-1) * weight[:,i].unsqueeze(-1)
                prediction_prob += facet_lm_logits_softmax * weight[:,:,i].unsqueeze(-1)
            elif self.weight_mode == 'static':
                #prediction_prob += facet_lm_logits_real_arr[i][..., :-1, :].contiguous().view(-1, self.vocab_size).softmax(dim=-1) * weight[i]
                prediction_prob += facet_lm_logits_softmax * weight[i]
            else:
                #prediction_prob += facet_lm_logits_real_arr[i][..., :-1, :].contiguous().view(-1, self.vocab_size).softmax(dim=-1) / n_facet_effective
                prediction_prob += facet_lm_logits_softmax / n_facet_effective

        outputs = (prediction_prob,) + (stacked_facet_lm_logits, ) + transformer_outputs[1:]
        #outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:
            # Shift so that tokens < n predict n
            #shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels_flat = shift_labels.view(-1)
            shift_prediction_prob = prediction_prob[..., :-1, :].contiguous()
            # Flatten the tokens

            loss_fct = torch.nn.NLLLoss(reduction='none')
            #loss = loss_fct(torch.log(shift_prediction_prob.view(-1, self.vocab_size)+1e-8),shift_labels.view(-1))
            loss_raw = loss_fct(torch.log(shift_prediction_prob.view(-1, self.vocab_size)+1e-8), shift_labels_flat)
            loss = loss_raw[shift_labels_flat != -100].mean()
            #loss = loss_raw.mean()
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
                    if n_facet_effective < self.n_facet:
                        if self.efficient_mode == "even":
                            n_facet, bsz, seq_len, emb_size = stacked_facet_emb.size()
                            stacked_facet_emb_n = torch.zeros( (n_facet_effective, bsz, seq_len, emb_size), dtype=torch.float, device = stacked_facet_emb.device)
                            stacked_facet_emb_n[0,:,:,:] = stacked_facet_emb[:, :, :, :].mean(dim=0)
                        else:
                            assert self.efficient_mode == "even_last_2" or self.efficient_mode == "even_last"
                            n_facet_partition = self.n_facet - n_facet_effective + 1
                            n_facet, bsz, seq_len, emb_size = stacked_facet_emb.size()
                            stacked_facet_emb_n = torch.zeros( (n_facet_effective, bsz, seq_len, emb_size), dtype=torch.float, device = stacked_facet_emb.device)
                            stacked_facet_emb_n[1:, :, :, :] = stacked_facet_emb[n_facet_partition:, :, :, :]
                            stacked_facet_emb_n[0,:,:,:] = stacked_facet_emb[:n_facet_partition, :, :, :].mean(dim=0)
                    else:
                        stacked_facet_emb_n = stacked_facet_emb
                    #assert self.efficient_mode == "None"
                    #pred_mean = stacked_facet_emb.mean(dim = 0, keepdim = True) # facet, bsz, seq_len, emb_size
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
                        #temp = collapse_difference.min()
                        #temp = collapse_difference.min()
                        #temp_idx = collapse_difference.argmin()
                        #if temp.item() < 0.1:
                        #    print(temp)
                        #    print(collapse_difference.view(-1)[temp_idx])
                        #    print(top_idx_single.view(-1,20)[temp_idx,:])
                        #    print(top_idx_before.view(-1,20)[temp_idx,:])
                        #    print(top_val_now.view(-1,20)[temp_idx,:])
                        #    print(top_val_before.view(-1,20)[temp_idx,:])
                        #    print(prediction_prob.view(-1,50257)[temp_idx,:])
                        #    print(collapse_single_prob.view(-1,50257)[temp_idx,:])
                        #    print(pred_mean_weighted.view(-1,768)[temp_idx,:])
                        #    print(stacked_facet_emb_n.view(3,-1,768)[:,temp_idx,:])
                        #    sys.exit()
                    else:
                        facet_norm = projected_emb_arr[0].norm(dim=-1)


            #loss_fct = CrossEntropyLoss(ignore_index = -100)
            #loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
            #best_idx = torch.gather(lm_max_idx[..., :-1, :], dim=-1 , index = shift_labels.unsqueeze(dim=-1))
            if not self.only_commpute_loss:
                with torch.no_grad():
                    lm_logits_max, lm_max_idx = torch.max(stacked_facet_lm_logits.softmax(dim=-1), dim=0)
                    count_best_arr = torch.zeros( (1, n_facet_effective), device = lm_max_idx.device)
                    shift_labels = input_ids[..., 1:].contiguous()
                    best_idx = torch.gather(lm_max_idx[..., :-1, :], dim=-1 , index = shift_labels.unsqueeze(dim=-1))
                    have_label_best_idx = best_idx.squeeze(dim=-1)
                    for i in range(n_facet_effective):
                        #count = torch.sum(have_label_best_idx == i).item()
                        #count_best_arr.append(count)
                        count = torch.sum(have_label_best_idx == i)
                        count_best_arr[0,i] = count
                    #if self.training:
                    if False:
                        weight_avg = weight.mean(dim=0).mean(dim=0)
                        min_idx = torch.argmin(weight_avg)
                        min_val = weight_avg[min_idx]
                        if min_val < 0.01:
                            self.starve_idx = min_idx
                    #if self.training:
                    if False:
                        #min_idx = np.argmin(count_best_arr)
                        min_idx = torch.argmin(count_best_arr[0,:])
                        min_val = count_best_arr[0,min_idx]
                        #if min_val < 10:
                        #if False:
                        if min_val == 0:
                            #if random.random() < 0.1:
                            if True:
                            #if False:
                                self.starve_idx = min_idx
                
        if self.only_commpute_loss:
            return outputs
        elif vis_simple:
            return outputs, emb_div, count_best_arr, weight, emb_div_arr_vis.view(1,-1), loss_seq_vis.view(1,-1), MRR_seq_vis.view(1,-1), loss_raw, MRR_raw, div_raw.mean(dim=0)

        elif vis_seq_loss:
            #return outputs, emb_div, count_arr, np.array(count_best_arr), emb_div_arr_vis.cpu().numpy(), loss_seq_vis.cpu().numpy()
            #return outputs, emb_div, count_arr, count_best_arr, emb_div_arr_vis.view(1,-1), loss_seq_vis.view(1,-1), MRR_seq_vis.view(1,-1), loss_raw, MRR_raw, dist_to_period.view(-1)
            return outputs, emb_div, count_best_arr, weight, emb_div_arr_vis.view(1,-1), loss_seq_vis.view(1,-1), MRR_seq_vis.view(1,-1), loss_raw, MRR_raw, div_raw.mean(dim=0), dist_to_period.view(-1), div_weighted_sq_raw, collapse_difference, collapse_difference_inv, collapse_difference_val, facet_norm, top_val_single, top_idx_single
        elif output_weight:
            #return outputs, emb_div, count_arr, np.array(count_best_arr)
            #return outputs, emb_div, count_arr, count_best_arr, weight
            return outputs, emb_div, count_best_arr, weight
        else:
            #return outputs, emb_div, count_arr, count_best_arr
            return outputs, emb_div, count_best_arr
        


class GPT2MultiLMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config, transformer_input, n_facet, n_facet_hidden, use_avg=False):
        super(GPT2MultiLMHeadModel, self).__init__(config)
        #self.transformer = GPT2Model(config)
        self.n_facet = n_facet
        if n_facet_hidden == 0:
            n_facet_hidden = n_facet
        self.project_arr = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=False) for i in range(n_facet)])
        #self.project_arr[0].weight.data = torch.eye(config.n_embd)
        small_value = 0.0001
        num_init_facet = n_facet
        if n_facet == 3:
            num_init_facet = 2
        for i in range(n_facet):
            if i <num_init_facet:
                self.project_arr[i].weight.data = torch.eye(config.n_embd) + small_value * (torch.rand((config.n_embd, config.n_embd)) - 0.5 )
            else:
                self.project_arr[i].weight.data = small_value * (torch.rand((config.n_embd, config.n_embd)) - 0.5 )
        #for i in range(num_init_facet):
        #    self.project_arr[i].weight.data = torch.eye(config.n_embd) + small_value * (torch.rand((config.n_embd, config.n_embd)) - 0.5 )
        #self.multi_project = nn.Linear(config.n_embd, config.n_embd*n_facet)
        #self.project_arr = nn.Linear(config.n_embd, config.n_embd*n_facet_hidden)
        #self.project_arr_2 = nn.Linear(config.n_embd*n_facet_hidden, config.n_embd*n_facet_hidden)
        #self.project_arr_3 = nn.Linear(config.n_embd*n_facet_hidden, config.n_embd*n_facet)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #self.init_weights()
        self.transformer = transformer_input
        self.vocab_size = config.vocab_size
        self.use_avg = use_avg
        self.output_probs = False
        self.starve_idx = -1

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        #hidden_states = transformer_outputs[0]
        all_hidden_states = transformer_outputs[2]
        ##projected_emb = gelu(self.project_arr(hidden_states))
        ##projected_emb = gelu(self.project_arr_2(projected_emb))
        ##projected_emb = self.project_arr_3(projected_emb)
        #projected_emb = self.multi_project(hidden_states)
        #bsz, max_seq_len, emb_size_n_facet = projected_emb.size()
        #stacked_facet_emb = projected_emb.view(bsz, max_seq_len,self.n_facet,-1)
        #stacked_facet_lm_logits = self.lm_head(stacked_facet_emb)
        #lm_logits, lm_max_idx = torch.max(stacked_facet_lm_logits, dim=2)
        #lm_logits = torch.log(torch.sum(torch.exp(stacked_facet_lm_logits),dim=2))

        #stacked_facet_emb_mean = stacked_facet_emb.mean(dim=2)
        #lm_logits = (lm_logits + self.lm_head(stacked_facet_emb_mean) ) /2
        #with torch.no_grad():
        #    stacked_facet_emb = stacked_facet_emb / (1e-12 + stacked_facet_emb.norm(dim = -1, keepdim=True))
        #    pred_mean = stacked_facet_emb.mean(dim = 2, keepdim = True)
        #    emb_div = - torch.mean( (stacked_facet_emb - pred_mean).norm(dim = -1) )
        #facet_lm_logits_arr = []

        facet_lm_logits_real_arr = []
        facet_lm_logits_arr = []
        projected_emb_real_arr = []
        projected_emb_arr = []
        for i in range(self.n_facet):
            #hidden_states = all_hidden_states[-(i+1)]
            hidden_states = all_hidden_states[-1]
            projected_emb = self.project_arr[i](hidden_states)
            projected_emb_arr.append(projected_emb)
            facet_lm_logits_arr.append( self.lm_head(projected_emb) )
            if self.starve_idx < 0 or self.starve_idx == i:
                facet_lm_logits_real_arr.append( facet_lm_logits_arr[-1] )
                projected_emb_real_arr.append(projected_emb)
            else:
                bsz, seq_len, hidden_size = all_hidden_states[-1].size()
                facet_lm_logits_real_arr.append(torch.zeros( (bsz, seq_len, self.vocab_size), device=all_hidden_states[-1].device ))
        self.starve_idx = -1
        stacked_facet_lm_logits = torch.stack(facet_lm_logits_arr, dim=0)
        stacked_facet_lm_real_logits = torch.stack(facet_lm_logits_real_arr, dim=0)
        lm_logits_max, lm_max_idx = torch.max(stacked_facet_lm_logits, dim=0)
        lm_logits = torch.logsumexp(stacked_facet_lm_real_logits, dim=0)
        #lm_logits_log = torch.sum(torch.exp(stacked_facet_lm_logits),dim=0)
        #lm_logits = torch.log(torch.max(lm_logits_log,other=torch.ones_like(lm_logits_log)*1e-5))
        #lm_logits = 0
        
        if self.use_avg:
            stacked_facet_real_emb = torch.stack(projected_emb_real_arr, dim=0)
            stacked_facet_emb_real_mean = stacked_facet_real_emb.mean(dim=0)
            lm_logits = (lm_logits + self.lm_head(stacked_facet_emb_real_mean) ) /2

        with torch.no_grad():
            stacked_facet_emb = torch.stack(projected_emb_arr, dim=0)
            stacked_facet_emb = stacked_facet_emb / (1e-12 + stacked_facet_emb.norm(dim = -1, keepdim=True))
            pred_mean = stacked_facet_emb.mean(dim = 0, keepdim = True)
            emb_div = - torch.mean( (stacked_facet_emb - pred_mean).norm(dim = -1) )
            top_k = 100
            #count_arr = []
            count_arr = np.zeros(self.n_facet)
            for i in range(self.n_facet):
                count = torch.sum(lm_max_idx[:,:,:top_k] == i).item()
                count_arr[i] = count
                #count_arr.append(count)

        #self.multi_project(hidden_states)
        #lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + (stacked_facet_lm_logits, ) + transformer_outputs[1:]
        #outputs = (lm_logits,) + transformer_outputs[1:]
        count_best_arr = []
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index = -100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
            best_idx = torch.gather(lm_max_idx[..., :-1, :], dim=-1 , index = shift_labels.unsqueeze(dim=-1))
            have_label_best_idx = best_idx.squeeze(dim=-1)
            for i in range(self.n_facet):
                count = torch.sum(have_label_best_idx == i).item()
                count_best_arr.append(count)
            if self.training:
            #if False:
                min_idx = np.argmin(count_best_arr)
                min_val = count_best_arr[min_idx]
                if min_val == 0:
                    #if random.random() < 0.1:
                    if True:
                        self.starve_idx = min_idx
        return outputs, emb_div, count_arr, np.array(count_best_arr)
        # (loss), lm_logits, stacked_facet_lm_logits , presents, (all hidden_states), (attentions)
        # (loss), lm_logits, presents, (all hidden_states), (attentions)


@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling head on top
(linear layer with weights tied to the input embeddings). """,
    GPT2_START_DOCSTRING,
    GPT2_INPUTS_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(2, batch_size, num_heads, sequence_length, embed_size_per_head)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index = -100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling and a multiple-choice classification
head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
The language modeling head has its weights tied to the input embeddings,
the classification head takes as input the input of a specified classification token index in the input sequence).
""",
    GPT2_START_DOCSTRING,
    GPT2_INPUTS_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    r"""
        **mc_token_ids**: (`optional`, default to index of the last token of the input) ``torch.LongTensor`` of shape ``(batch_size, num_choices)``:
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        **mc_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **mc_loss**: (`optional`, returned when ``multiple_choice_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Multiple choice classification loss.
        **lm_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **mc_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)``
            Prediction scores of the multiplechoice classification head (scores for each choice before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(2, batch_size, num_heads, sequence_length, embed_size_per_head)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

    """

    def __init__(self, config):
        super(GPT2DoubleHeadsModel, self).__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)
