#!/usr/bin/env python

import json
import sys
import argparse
from collections import OrderedDict
from __init__ import *
from config import constant 

def parse() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_freq', default=None, type=int)
    parser.add_argument('--max_vocab', default=None, type=int)
    parser.add_argument('--dict', type=str)
    parser.add_argument('--freq', type=str)
    return parser.parse_args()

if __name__ == '__main__' :
    args = parse()
    original_dict = OrderedDict(json.load(open(args.dict)))
    original_freq = OrderedDict(json.load(open(args.freq)))
    assert args.min_freq != args.max_vocab, "user should only provide one of them"
    for w1, w2 in zip(original_dict.keys(), original_freq.keys()) :
        assert w1 == w2, "word order is not same !"
    included_word = 0
    total_word = 0
    if args.max_vocab :
        sorted_items = sorted(original_dict.items(), key=lambda x : x[1])
        sorted_items = sorted_items[0:args.max_vocab]
        modified_dict = OrderedDict(sorted_items)
        json.dump(modified_dict, open(args.dict+'.maxv{}'.format(args.max_vocab), 'w'), indent=2, ensure_ascii=False)
    else :
        filtered_word = set()
        for ww, ff in original_freq.items() :
            if ff >= args.min_freq :
                filtered_word.add(ww)
        sorted_items = sorted(original_dict.items(), key=lambda x : x[1])
        sorted_items = [(x, y) for (x, y) in sorted_items if x in filtered_word]
        modified_dict = OrderedDict(sorted_items)
        json.dump(modified_dict, open(args.dict+'.minf{}'.format(args.min_freq), 'w'), indent=2, ensure_ascii=False)

    included_word = sum([original_freq[x] for x in modified_dict.keys() if x not in [constant.BOS_WORD, constant.EOS_WORD, constant.UNK_WORD, constant.PAD_WORD]])
    total_word = sum([y for (x,y) in original_freq.items() if x not in [constant.BOS_WORD, constant.EOS_WORD, constant.UNK_WORD, constant.PAD_WORD]])
    print('[INFO] ratio in/total : {}/{} \t( {:.3f} \% )'.format(included_word, total_word, included_word/total_word * 100))
    pass
