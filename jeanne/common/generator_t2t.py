"""
generator_s2t.py

generator for ASR model (speech -> text)
"""
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from tamamo.utils.serializer import ModelSerializer
from tamamo.utils.helper import tensorauto, torchauto, vars_index_select
from ..config import constant

def greedy_search(model, src_mat, src_len, map_text2idx, max_target, aux_info={'type':'basic'}) :
    transcription, transcription_len, att_mat = greedy_search_torch(model, 
            src_mat, src_len, map_text2idx, max_target, aux_info=aux_info)
    return convert_transcription_to_list(transcription, transcription_len), convert_att_to_list(att_mat, transcription_len)

def greedy_search_torch(model, src_mat, src_len, map_text2idx, max_target, aux_info={'type':'asr'}) :
    batch = src_mat.size(0)
    model.reset()
    if aux_info['type'] in ['nmt', 'asr'] :
        model.encode(Variable(src_mat), src_len)
    else :
        raise NotImplementedError

    prev_label = Variable(torchauto(model).LongTensor([map_text2idx[constant.BOS_WORD] for _ in range(batch)]))
    transcription = []
    transcription_len = [-1 for _ in range(batch)]
    att_mat = []
    for tt in range(max_target) :
        pre_softmax, dec_res = model.decode(prev_label)
        max_id = pre_softmax.max(1)[1]
        transcription.append(max_id)
        att_mat.append(dec_res['att_output']['p_ctx'])
        for ii in range(batch) :
            if transcription_len[ii] == -1 :
                if max_id.data[ii] == map_text2idx[constant.EOS_WORD] :
                    transcription_len[ii] = tt+1
        if all([ii != -1 for ii in transcription_len]) :
            # finish #
            break
        prev_label = max_id
        pass

    # concat across all timestep #
    transcription = torch.stack(transcription, 1) # batch x seq_len #
    att_mat = torch.stack(att_mat, 1) # batch x seq_len x enc_len #
    
    return transcription, transcription_len, att_mat
    pass

def convert_transcription_to_list(transcription, transcription_len) :
    result = []
    for ii in range(len(transcription_len)) :
        curr_result = transcription[ii].data.cpu().numpy().tolist()
        if transcription_len[ii] != -1 :
            curr_result = curr_result[0:transcription_len[ii]-1] # remove <eos> #
        result.append(curr_result)
    return result

def convert_att_to_list(att_mat, transcription_len) :
    result = []
    for ii in range(len(transcription_len)) :
        curr_result = att_mat[ii].data.cpu().numpy().tolist()
        if transcription_len[ii] != -1 :
            curr_result = curr_result[0:transcription_len[ii]-1] # remove <bos> and <eos> #
        result.append(curr_result)
    return result
