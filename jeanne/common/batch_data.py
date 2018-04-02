from __init__ import *
import torch
import numpy as np
from tamamo.utils.helper import tensorauto, torchauto
from tamamo.utils.mask_util import generate_seq_mask
from ..config import constant
import librosa

def extract_speech_feat(wav_list) :
    batch = len(wav_list)
    feat_list = []
    # load into mfcc
    for ii in range(batch) :
        _wav, _sr = librosa.load(wav_list[ii], sr=None)
        feat_list.append(librosa.feature.mfcc(y=_wav, sr=_sr, n_mfcc=constant.N_MEL).T)
    feat_len = [len(x) for x in feat_list]
    return feat_list, feat_len
    pass

def batch_speech(device, feat_list, feat_len) :
    batch = len(feat_list)
    max_feat_len = max(feat_len)
    feat_mat = np.zeros((batch, max_feat_len, feat_list[0].shape[1]), dtype='float32')
    for ii in range(batch) :
        feat_mat[ii, 0:feat_len[ii]] = feat_list[ii]
    feat_mat = tensorauto(device, torch.from_numpy(feat_mat))
    return feat_mat, feat_len

def batch_sents(device, text_list, add_sos=True, add_eos=True) :
    batch = len(text_list)
    text_idx_list = (text_list)
    # add sos & eos #
    if add_sos :
        text_idx_list = [[constant.BOS]+x for x in text_idx_list]
    if add_eos :
        text_idx_list = [x+[constant.EOS] for x in text_idx_list]
    
    text_len = [len(x) for x in text_idx_list]
    text_mat = np.full((batch, max(text_len)), constant.PAD, dtype='int64')
    for ii in range(batch) :
        text_mat[ii, 0:text_len[ii]] = text_idx_list[ii]
    text_mat = tensorauto(device, torch.from_numpy(text_mat))
    return text_mat, text_len
    pass

def batch_src_tgt_sents(device, src_text_list, tgt_text_list) :
    src_mat, src_len = batch_sents(device, src_text_list)
    tgt_mat, tgt_len = batch_sents(device, tgt_text_list)
    return src_mat, src_len, tgt_mat, tgt_len

# ADDITIONAL #
def batch_scores(device, asr_score, add_sos=True, add_eos=True) :
    if add_sos :
        asr_score = [[constant.ASR_MAX_SCORE]+x for x in asr_score] 
    if add_eos :
        asr_score = [x+[constant.ASR_MAX_SCORE] for x in asr_score]
    batch = len(asr_score)
    len_score = [len(x) for x in asr_score]
    asr_score_mat = np.full((batch, max(len_score)), 0, dtype='float32')
    for ii in range(batch) :
        #import pdb; pdb.set_trace() 
        asr_score_mat[ii, 0:len_score[ii]] = asr_score[ii]
    asr_score_mat = tensorauto(device, torch.from_numpy(asr_score_mat))
    return asr_score_mat
    pass


def text2idx(sents, map_text2idx) :
    result = []
    for sent in sents :
        words = sent.strip().split()
        words_id = []
        for word in words :
            if word in map_text2idx :
                words_id.append(map_text2idx[word])
            else :
                words_id.append(constant.UNK)
        result.append(words_id)
        pass
    return result

def keytext2idx(key_sents, map_text2idx) :
    result = []
    _keys = [x[0] for x in key_sents]
    _values = [x[1] for x in key_sents]
    _values = text2idx(_values, map_text2idx)
    return list(zip(_keys, _values))

def idx2text(sents_idx, map_idx2text) :
    result = []
    for sent in sents_idx :
        result.append(' '.join([map_idx2text[word] for word in sent]))
    return result
