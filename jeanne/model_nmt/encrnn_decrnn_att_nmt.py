
import sys
import time
import re

import numpy as np
import json

# pytorch #
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

# torchev #
from tamamo.generator import generator_rnn, generator_attention
from tamamo.custom import decoder
from tamamo.utils.helper import torchauto

# utilbox #
from .utils.config_util import ConfigParser

class ENCRNN_DECRNN_ATT_NMT(nn.Module) :
    def __init__(self, enc_in_size, dec_in_size,
            enc_emb_size=256, enc_emb_do=0.0,
            dec_emb_size=256, dec_emb_do=0.0,
            enc_rnn_sizes=[512, 512], enc_rnn_cfgs={"type":"lstm", "bi":True}, enc_rnn_do=0.25,
            dec_rnn_sizes=[512, 512], dec_rnn_cfgs={"type":"lstm"}, dec_rnn_do=0.25,
            dec_cfg={"type":"standard_decoder"}, 
            att_cfg={"type":"mlp"},
            ) :
        super(ENCRNN_DECRNN_ATT_NMT, self).__init__()

        self.enc_emb_size = enc_emb_size
        self.enc_emb_do = enc_emb_do
        self.dec_emb_size = dec_emb_size
        self.dec_emb_do = dec_emb_do

        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.enc_rnn_sizes = enc_rnn_sizes
        self.enc_rnn_cfgs = enc_rnn_cfgs
        self.enc_rnn_do =  ConfigParser.list_parser(enc_rnn_do, len(enc_rnn_sizes))

        self.dec_rnn_sizes = dec_rnn_sizes
        self.dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes)) 
        self.dec_rnn_do = ConfigParser.list_parser(dec_rnn_do, len(dec_rnn_sizes))
        self.dec_cfg = dec_cfg
        self.att_cfg = att_cfg

        # modules #
        # init encoder #
        self.enc_emb = nn.Embedding(self.enc_in_size, self.enc_emb_size, padding_idx=None)
        prev_size = enc_emb_size
        self.enc_rnn = nn.ModuleList()
        _enc_rnn_cfgs = ConfigParser.list_parser(enc_rnn_cfgs, len(enc_rnn_sizes))
        for ii in range(len(enc_rnn_sizes)) :
            _rnn_cfg = {}
            _rnn_cfg['type'] = _enc_rnn_cfgs[ii]['type']
            _rnn_cfg['args'] = [prev_size, enc_rnn_sizes[ii], 1, True, True, 0, _enc_rnn_cfgs[ii]['bi']]
            self.enc_rnn.append(generator_rnn(_rnn_cfg))
            prev_size = enc_rnn_sizes[ii] * (2 if _enc_rnn_cfgs[ii]['bi'] else 1)
        final_enc_size = prev_size

        # init decoder #
        self.dec_emb = nn.Embedding(self.dec_in_size, self.dec_emb_size, padding_idx=None)
        prev_size = dec_emb_size
        _dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes))
        for ii in range(len(dec_rnn_sizes)) :
            _type = _dec_rnn_cfgs[ii]['type']
            if re.match('stateful.*cell', _type) is None :
                _dec_rnn_cfgs[ii]['type'] = 'stateful_{}cell'.format(_type)
        # TODO : dec_cfg #
        self.dec = decoder.StandardDecoder(att_cfg, final_enc_size, self.dec_emb_size, 
                dec_rnn_sizes, _dec_rnn_cfgs, dec_rnn_do)
        self.pre_softmax = nn.Linear(self.dec.output_size, self.dec_in_size)
        pass 

    def get_config(self) :
        # TODO
        return {'class':str(self.__class__),
                'enc_in_size':self.enc_in_size,
                'dec_in_size':self.dec_in_size,
                'enc_emb_size':self.enc_emb_size,
                'enc_emb_do':self.enc_emb_do,
                'enc_rnn_sizes':self.enc_rnn_sizes,
                'enc_rnn_cfgs':self.enc_rnn_cfgs,
                'enc_rnn_do':self.enc_rnn_do,
                'dec_emb_size':self.dec_emb_size,
                'dec_emb_do':self.dec_emb_do,
                'dec_rnn_sizes':self.dec_rnn_sizes,
                'dec_rnn_cfgs':self.dec_rnn_cfgs,
                'dec_rnn_do':self.dec_rnn_do,
                'dec_cfg':self.dec_cfg,
                'att_cfg':self.att_cfg,
                }
        pass

    def encode(self, input, src_len=None) :
        """
        input : (batch x max_src_len) OR (batch x max_src_len x vocab_dim)
        mask : (batch x max_src_len)
        """
        batch, max_src_len = input.size()[0:2]
        
        if src_len is None :
            src_len = [max_src_len] * batch
        # MODIFIED #
        if input.dim() == 2 :
            res = self.enc_emb(input)
        else :
            # WARNING : speed bottleneck #
            res = torch.mm(input.view(-1, self.enc_in_size), self.enc_emb.weight)\
                    .view(batch, max_src_len, -1) # backprop-able wrt input #

        res = F.dropout(res, self.enc_emb_do, self.training)

        for ii in range(len(self.enc_rnn)) :
            res = pack(res, src_len, batch_first=True)
            res = self.enc_rnn[ii](res)[0] # get h only #
            res,_ = unpack(res, batch_first=True)
            res = F.dropout(res, self.enc_rnn_do[ii], self.training)
        ctx = res
        # create mask if required #
        if src_len is not None :
            ctx_mask = torchauto(self).FloatTensor(batch, ctx.size(1)).zero_()
            for ii in range(batch) :
                ctx_mask[ii, 0:src_len[ii]] = 1.0
            ctx_mask = Variable(ctx_mask)
        else : 
            ctx_mask = None
        self.dec.set_ctx(ctx, ctx_mask)

    def reset(self) :
        self.dec.reset()

    def decode(self, y_tm1, mask=None) :
        assert y_tm1.dim() == 1, "batchsize only"
        res = self.dec_emb(y_tm1)
        if self.dec_emb_do > 0.0 :
            res = F.dropout(res, self.dec_emb_do, self.training)
        res = self.dec(res, mask)
        return self.pre_softmax(res['dec_output']), res 

    @property
    def state(self) :
        return (self.dec.state, )

    @state.setter
    def state(self, value) :
        self.dec.state = value[0]
    
