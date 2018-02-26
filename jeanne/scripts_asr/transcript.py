import argparse
import os
import json
import sys
import logging
import multiprocessing
import operator
from tqdm import tqdm
def tqdm_wrapper(obj) :
    return tqdm(obj, ascii=True, ncols=50)

from __init__ import *
from jeanne.config import constant
from jeanne.common.generator_t2t import greedy_search
from jeanne.common.batch_data import text2idx, idx2text, batch_sents, batch_speech, extract_speech_feat

import pickle
from scipy.io import wavfile
import numpy as np
import warnings
import time
import torch
from torch.autograd import Variable
from jeanne.util.regex_util import regex_key_val
from tamamo.utils.serializer import ModelSerializer
from tamamo.utils.helper import tensorauto, torchauto

"""
translate.py

script for translate sentences given trained model (cpu or gpu)
"""
def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model', type=str, help='model weight')
    parser.add_argument('--data_cfg', type=str, help='data config (retrieve vocab map)')
    parser.add_argument('--config', type=str, help='model config', default=None)
    parser.add_argument('--scp', type=str, required=True, help='scp file contains key & wav path')
    parser.add_argument('--chunk', type=int, default=10, help='chunk size for each iterative')
    parser.add_argument('--max_target', type=int, default=200, help='max target for producing hypothesis')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id for decoding process (-1 for cpu)')
    # TODO search type (beam, greedy) #
    parser.add_argument('--search', type=str, default='greedy', choices=['greedy'])
    return parser.parse_args()

if __name__ == '__main__' :
    opts = vars(parse())
    # init logger #
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # load model structure and weight #
    if opts['config'] is None :
        opts['config'] = os.path.join(os.path.dirname(opts['model']), 'model.cfg')
    model = ModelSerializer.load_config(opts['config'])
    model.train(False)
    model.load_state_dict(torch.load(opts['model']))

    if opts['gpu'] >= 0 :
        torch.cuda.set_device(opts['gpu'])
        model.cuda()
    
    data_cfg = json.load(open(opts['data_cfg']))

    tgt_vocab2idx = json.load(open(data_cfg['tgt']['vocab']))
    tgt_idx2vocab = dict([(y, x) for (x, y) in tgt_vocab2idx.items()])
    
    feat_list_kv = list(map(list, regex_key_val.findall(open(opts['scp']).read())))
    feat_key, feat_list = list(zip(*feat_list_kv))
    feat_list, feat_len = extract_speech_feat(feat_list)
    # sort by length #
    sorted_feat_idx = np.argsort(feat_len).tolist()[::-1] # reversed
    sorted_feat_len = operator.itemgetter(*sorted_feat_idx)(feat_len)

    # decode with single CPU TODO : multiprocessing #
    start = time.time()
    best_hypothesis = []
    for ii in tqdm_wrapper(range(0, len(feat_list_kv), opts['chunk'])) :
        rr = [sorted_feat_idx[_ii] for _ii in list(range(ii, min(ii+opts['chunk'], len(feat_list_kv))))]
        curr_src_list = [feat_list[rrii] for rrii in rr]
        curr_src_len_list = [len(x) for x in curr_src_list]
        src_mat, src_len = batch_speech(opts['gpu'], curr_src_list, curr_src_len_list)
        if opts['search'] == 'greedy' :
            curr_best_hypothesis, curr_best_att = greedy_search(model, src_mat, src_len, tgt_vocab2idx, opts['max_target'], aux_info={'type':'asr'})
            best_hypothesis.extend(curr_best_hypothesis)
        else :
            raise ValueError('search method is not defined')
        pass
    # inverse to original position #
    best_hypothesis = [best_hypothesis[ii] for ii in np.argsort(sorted_feat_idx)]
    # map best hypothesis into word #
    best_hypothesis = idx2text(best_hypothesis, tgt_idx2vocab)
    
    # print best hypothesis #
    for kk, ss in zip(feat_key, best_hypothesis) :
        print(kk+' '+ss)
    logger.info('Finished transcript {} speech signals'.format(len(best_hypothesis)))
    pass
