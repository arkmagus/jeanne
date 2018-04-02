import os
import sys

import argparse
import json
import numpy as np
import itertools
import time
import operator
from tqdm import tqdm
def tqdm_wrapper(obj) :
    return tqdm(obj, ascii=True, ncols=50)
import tabulate as tab

# pytorch #
import torch
from torch import nn
from torch.autograd import Variable
from tamamo.utils.serializer import ModelSerializer
from tamamo.nn.modules.loss_elementwise import ElementwiseCrossEntropy
from tamamo.utils.helper import tensorauto, torchauto

# utilbox #
from jeanne.util.data_util import iter_minibatches, iter_minibucket
from jeanne.util.math_util import assert_nan
from jeanne.util.log_util import logger_stdout_file
from jeanne.util.regex_util import regex_key_val
from jeanne.model_asr.encrnn_decrnn_att_asr import ENCRNN_DECRNN_ATT_ASR
from jeanne.common.batch_data import keytext2idx, batch_speech, batch_sents, extract_speech_feat
from jeanne.config import constant

DEBUG = False
def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--dec_emb_size', type=int, default=256)
    parser.add_argument('--dec_emb_do', type=float, default=0.0)

    parser.add_argument('--enc_rnn_sizes', type=int, default=[512, 512], nargs='+')
    parser.add_argument('--enc_rnn_cfgs', type=json.loads, default='{"type":"lstm", "bi":true}', help='rnn connection type')
    parser.add_argument('--enc_rnn_do', type=float, nargs='+', default=[0.25, 0.25])
    
    parser.add_argument('--dec_rnn_sizes', type=int, default=[512, 512], nargs='+')
    parser.add_argument('--dec_rnn_cfgs', type=json.loads, default='{"type":"lstm"}', help='rnn connection type')
    parser.add_argument('--dec_rnn_do', type=float, nargs='+', default=[0.25, 0.25])

    parser.add_argument('--dec_cfg', type=json.loads, default='{"type":"standard_decoder"}', help='decoder type')
    parser.add_argument('--att_cfg', type=json.loads, default='{"type":"mlp"}', help='attention type')
    
    parser.add_argument('--data_cfg', type=str, default='config/dataset_wsj.json')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=15)

    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--lrate', type=float, default=0.001) # default is not always converge
    parser.add_argument('--decay', type=float, default=1.0, help='decay lrate after no dev cost improvement')
    parser.add_argument('--grad_clip', type=float, default=20.0) # grad clip to prevent NaN
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--result', type=str, default='expr_seq2seq_asr/dummy')
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--sortagrad', type=int, default=-1)
    parser.add_argument('--cutoff', type=int, default=-1, help='cutoff frame larger than x')

    return parser.parse_args()

def loader_seq2seq(data_config) :
    
    # 1. get data #
    data = {'src':{}, 'tgt':{}}
    data['src']['train'] = list(map(list, regex_key_val.findall(open(data_config['src']['train']).read())))
    data['tgt']['train'] = list(map(list, regex_key_val.findall(open(data_config['tgt']['train']).read())))

    data['src']['dev'] = list(map(list, regex_key_val.findall(open(data_config['src']['dev']).read())))
    data['tgt']['dev'] = list(map(list, regex_key_val.findall(open(data_config['tgt']['dev']).read())))

    data['src']['test'] = list(map(list, regex_key_val.findall(open(data_config['src']['test']).read())))
    data['tgt']['test'] = list(map(list, regex_key_val.findall(open(data_config['tgt']['test']).read())))

    # replace path with feature #
    for set_name in ['train', 'dev', 'test'] :
        for ii in tqdm_wrapper(range(len(data['src'][set_name]))) :
            # get only the feature
            data['src'][set_name][ii][1] = extract_speech_feat([data['src'][set_name][ii][1]])[0][0]

    data['tgt']['vocab'] = json.load(open(data_config['tgt']['vocab']))

    # 2. convert wav to feature #
    
    # convert text to idx #
    data['tgt']['train'] = keytext2idx(data['tgt']['train'], data['tgt']['vocab'])
    data['tgt']['dev'] = keytext2idx(data['tgt']['dev'], data['tgt']['vocab'])
    data['tgt']['test'] = keytext2idx(data['tgt']['test'], data['tgt']['vocab'])
    return data

if __name__ == '__main__' :
    opts = vars(parse())
    print(opts)

    # set default device #
    np.random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    if opts['gpu'] >= 0 :
        torch.cuda.set_device(opts['gpu'])
        torch.cuda.manual_seed(opts['seed'])
    
    # dataset #
    data_list = loader_seq2seq(json.load(open(opts['data_cfg'])))
    
    print("Finish loading dataset ...")
    
    NDIM_SRC = constant.N_MEL
    NVOCAB_TGT = len(data_list['tgt']['vocab'])
    model = ENCRNN_DECRNN_ATT_ASR(
            enc_in_size=NDIM_SRC, dec_in_size=NVOCAB_TGT, n_class=NVOCAB_TGT,
            dec_emb_size=opts['dec_emb_size'], dec_emb_do=opts['dec_emb_do'],
            enc_rnn_sizes=opts['enc_rnn_sizes'],
            enc_rnn_cfgs=opts['enc_rnn_cfgs'],
            enc_rnn_do=opts['enc_rnn_do'],
            
            dec_rnn_sizes=opts['dec_rnn_sizes'],
            dec_rnn_cfgs=opts['dec_rnn_cfgs'],
            dec_rnn_do=opts['dec_rnn_do'],
            dec_cfg=opts['dec_cfg'],
            att_cfg=opts['att_cfg'],
            )

    crit_weight = tensorauto(opts['gpu'], torch.ones(NVOCAB_TGT))
    crit_weight[constant.PAD] = 0
    crit_weight = Variable(crit_weight, requires_grad=False)
    criterion = ElementwiseCrossEntropy(weight=crit_weight)

    if opts['gpu'] >= 0 :
        model.cuda(opts['gpu'])
        criterion.cuda()
        pass
    
    # setting optimizer #
    opt = getattr(torch.optim, opts['opt'])(model.parameters(), lr=opts['lrate'])

    def fn_batch(src_mat, src_len, tgt_mat, tgt_len, train_step=True) :
        # shift #
        tgt_input = tgt_mat[:, 0:-1]
        tgt_output = tgt_mat[:, 1:]

        src_mat, tgt_input, tgt_output = [Variable(x) for x in (src_mat, tgt_input, tgt_output)]
        model.reset() 
        model.train(train_step)
        model.encode(src_mat, src_len)
        batch, dec_len = tgt_output.size()
        list_pre_softmax = []
        #import pdb; pdb.set_trace()
        for ii in range(dec_len) :
            _pre_softmax_ii, _ = model.decode(tgt_input[:, ii])
            list_pre_softmax.append(_pre_softmax_ii)
            pass
        pre_softmax = torch.stack(list_pre_softmax, 1)
        # denominator = tgt_output.ne(constant.PAD).data.sum()
        denominator = Variable(torchauto(model).FloatTensor(tgt_len)-1)
        loss = criterion(pre_softmax.view(batch * dec_len, -1), tgt_output.contiguous().view(batch * dec_len)).view(batch, dec_len).sum(dim=1) / denominator
        loss = loss.mean()
        acc = torch.max(pre_softmax, 2)[1].squeeze(-1).data.eq(tgt_output.data).masked_select(tgt_output.ne(constant.PAD).data).sum() / denominator.sum()
        if train_step :
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            opt.step()

        return loss.data.sum(), acc.data.sum()
        pass
    
    N_TRAIN_SIZE = len(data_list['src']['train'])
    N_DEV_SIZE = len(data_list['src']['dev'])
    N_TEST_SIZE = len(data_list['src']['test'])
    train_key, train_feat_list = list(map(list, zip(*data_list['src']['train'])))
    dev_key, dev_feat_list = list(map(list, zip(*data_list['src']['dev'])))
    test_key, test_feat_list = list(map(list, zip(*data_list['src']['test'])))
    train_feat_len = [len(x) for x in train_feat_list]
    dev_feat_len = [len(x) for x in dev_feat_list]
    test_feat_len = [len(x) for x in test_feat_list]
    #### SORT BY LENGTH - SortaGrad ####
    sorted_train_idx = np.argsort(train_feat_len).tolist()
    sorted_dev_idx = np.argsort(dev_feat_len).tolist()
    sorted_test_idx = np.argsort(test_feat_len).tolist()
    
    sorted_train_len = operator.itemgetter(*sorted_train_idx)(train_feat_len)
    sorted_dev_len = operator.itemgetter(*sorted_dev_idx)(dev_feat_len)
    sorted_test_len = operator.itemgetter(*sorted_test_idx)(test_feat_len)

    # exclude cutoff #
    exc_idx = {}
    exc_idx['train'] = set([x for x in range(len(train_feat_len)) if train_feat_len[x] > opts['cutoff']])
    exc_idx['dev'] = set([x for x in range(len(dev_feat_len)) if dev_feat_len[x] > opts['cutoff']])
    exc_idx['test'] = set([x for x in range(len(test_feat_len)) if test_feat_len[x] > opts['cutoff']])
    
    def sort_reverse(idx, length) : # for encoder mask must be sorted decreasing #
        return sorted(idx, key=lambda x : length[x], reverse=True)
        pass
    ########################

    EPOCHS = opts['epoch']
    BATCHSIZE = opts['batchsize']
    
    # prepare model folder #
    if not os.path.exists(opts['result']) :
        os.makedirs(opts['result'])
    else :
        if len(os.listdir(opts['result'])) > 0:
            raise ValueError("Error : folder & data already existed !!!")
    
    print("=====START=====")
    prev_dev_loss = 2**64
    best_dev_loss = 2**64
    logger = logger_stdout_file(os.path.join(opts['result'], 'report.log'))
    for ee in range(EPOCHS) :
        start_time = time.time()
        mloss = dict(train=0, dev=0, test=0)
        macc = dict(train=0, dev=0, test=0)
        mcount = dict(train=0, dev=0, test=0)

        # choose standard training or bucket training #
        if opts['bucket'] :
            train_rr = iter_minibucket(sorted_train_idx, BATCHSIZE, 
                shuffle=False if ee < opts['sortagrad'] else True,
                excludes=exc_idx['train'])
            dev_rr = iter_minibucket(sorted_dev_idx, BATCHSIZE, 
                shuffle=False, excludes=exc_idx['train'])
            test_rr = iter_minibucket(sorted_test_idx, BATCHSIZE, 
                shuffle=False, excludes=exc_idx['train'])
        else :
            train_rr = iter_minibatches(sorted_train_idx, BATCHSIZE, 
                    shuffle=False if ee < opts['sortagrad'] else True, 
                    excludes=exc_idx['train'])
            dev_rr = iter_minibatches(sorted_dev_idx, BATCHSIZE, shuffle=False, 
                    excludes=exc_idx['train'])
            test_rr = iter_minibatches(sorted_test_idx, BATCHSIZE, shuffle=False, 
                    excludes=exc_idx['train'])
        ###############################################
        train_rr = [sort_reverse(x, train_feat_len) for x in train_rr] 
        dev_rr = [sort_reverse(x, dev_feat_len) for x in dev_rr] 
        test_rr = [sort_reverse(x, test_feat_len) for x in test_rr]

        for set_name, set_rr, set_train_mode in [('train', train_rr, True), ('dev', dev_rr, False), ('test', test_rr, False)] :
            for rr in tqdm_wrapper(set_rr) :
                curr_src_list = [data_list['src'][set_name][rrii][1] for rrii in rr]
                curr_src_len_list = [len(x) for x in curr_src_list]
                curr_tgt_list = [data_list['tgt'][set_name][rrii][1] for rrii in rr]
                src_mat, src_len = batch_speech(opts['gpu'], curr_src_list, curr_src_len_list)
                tgt_mat, tgt_len = batch_sents(opts['gpu'], curr_tgt_list) 
                _tmp_loss, _tmp_acc = fn_batch(src_mat, src_len, tgt_mat, tgt_len, train_step=set_train_mode)
                _tmp_count = len(rr)
                assert_nan(_tmp_loss)
                mloss[set_name] += _tmp_loss * _tmp_count
                macc[set_name] += _tmp_acc * _tmp_count
                mcount[set_name] += _tmp_count
            pass

        info_header = ['set', 'loss', 'acc']
        info_table = []

        logger.info("Epoch %d -- lrate %f --time %.2fs"%(ee+1, opt.param_groups[0]['lr'], time.time() - start_time))
        for set_name in mloss.keys() :
            mloss[set_name] /= mcount[set_name]
            macc[set_name] /= mcount[set_name]
            info_table.append([set_name, mloss[set_name], macc[set_name]])
        logger.info('\n'+tab.tabulate(info_table, headers=info_header, floatfmt='.3f', tablefmt='rst'))

        # serialized best dev model #
        if best_dev_loss > mloss['dev'] :
            best_dev_loss = mloss['dev'] 
            logger.info("\t# get best dev loss ... serialized the model")
            torch.save(ModelSerializer.convert_param_to_cpu(model.state_dict()), os.path.join(opts['result'], 'model.mdl'))
            ModelSerializer.save_config(os.path.join(opts['result'], 'model.cfg'), model.get_config())
            json.dump(opts, open(os.path.join(opts['result'], 'script.opts'), 'w'), indent=4)

        prev_dev_loss = mloss['dev']
        pass
    logger.info("best dev cost : %f"%(best_dev_loss))
    pass
