# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


'''
class NoamScheduler(_LRScheduler):
    
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Args:
        d_model: int
            The number of expected features in the encoder inputs.
        warmup_steps: int
            The number of steps to linearly increase the learning rate.
    
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

        # the initial learning rate is set as step = 1
        if self.last_epoch == -1:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
            self.last_epoch = 0


    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
'''


class SpeakerModel(nn.Module):
    def __init__(self, n_speakers, n_units):

        super(SpeakerModel, self).__init__()
        self.n_speakers = n_speakers
        # self.in_size = in_size
        # self.n_heads = n_heads
        self.n_units = n_units
        # self.n_layers = n_layers
        # self.has_pos = has_pos
        # self.one_mat = torch.ones([32, 500, 256], requires_grad=False).cuda().float()

        #  self.src_mask = None
        # self.encoder = nn.Linear(in_size, n_units)
        # self.encoder_norm = nn.LayerNorm(n_units)
        # if self.has_pos:
        #     self.pos_encoder = PositionalEncoding(n_units, dropout)
        # encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(n_units*2, n_units)
        self.activation = nn.Sigmoid()
        self.init_weights()
    
    '''
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    '''

    def init_weights(self):
        initrange = 0.1
        # self.encoder.bias.data.zero_()
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, embedding, labels, activation=None):
        '''
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        '''

        ilens = [x.shape[0] for x in embedding]
        src = nn.utils.rnn.pad_sequence(embedding, padding_value=-1, batch_first=True)
        
        lab = nn.utils.rnn.pad_sequence(labels, padding_value=-1, batch_first=True)
        # print("model start : ", src.shape)
        # print(lab.shape)
        # src: (B, T, E)
        # src = self.encoder(src)
        # src = self.encoder_norm(src)
        # src: (T, B, E)
        # src = src.transpose(0, 1)
        # if self.has_pos:
            # src: (T, B, E)
        #    src = self.pos_encoder(src)
        # output: (T, B, E)
        # output = self.transformer_encoder(src, self.src_mask)
        # output: (B, T, E)
        # output = output.transpose(0, 1)
        # output: (B, T, C)
        # frame_embedding = output.clone()
        
        spk_embedding = torch.matmul(lab.transpose(1, 2), src)  # [B, N_spk, embedding_dim]
        
        ones_mat = torch.ones(src.shape).float().cuda()
        cnt_mat = torch.matmul(lab.transpose(1, 2), ones_mat)
        cnt_mat[cnt_mat < 1.0] = 1.0
        spk_embedding_mean = spk_embedding / cnt_mat
        
        spk_embedding_square = torch.matmul(lab.transpose(1, 2), torch.square(src))
        spk_embedding_square_mean = spk_embedding_square / cnt_mat

        
        spk_embedding_var = spk_embedding_square_mean - torch.square(spk_embedding_mean)
        # output = self.decoder(spk_embedding)

        concated_spk_embedding = torch.cat((spk_embedding_mean, spk_embedding_var), dim=2)
        output = self.decoder(concated_spk_embedding)
        # output = self.decoder(spk_embedding)

        
        # output = [out[:ilen] for out, ilen in zip(output, ilens)]
        # output_act = output

        if activation:  # inference
            output = activation(output)

        # frame_embedding = [out[:ilen] for out, ilen in zip(frame_embedding, ilens)]
        output = [out for out, ilen in zip(output, ilens)]

        return output  # frame-wise embedding, output value(posteriors)

    '''
    def get_attention_weight(self, src):
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []
        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1])
            
        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward(src)

        for handle in handles:
            handle.remove()
        self.train()

        return torch.stack(attn_weight)
    '''
    
# class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    '''
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    '''

if __name__ == "__main__":
    import torch
    model = TransformerModel(5, 40, 4, 512, 2, 0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input).size())
    print("Model attention:", model.get_attention_weight(input).size())
    print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
