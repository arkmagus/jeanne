#!/usr/bin/env python

import numpy
import json

import sys
import fileinput
from __init__ import *
from config import constant
from collections import OrderedDict
MAXVAL = 10**15
def main():
    for filename in sys.argv[1:]:
        print('Processing ' +filename)
        dict_freq = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in dict_freq:
                        dict_freq[w] = 0
                    dict_freq[w] += 1
        words = list(dict_freq.keys())
        freqs = list(dict_freq.values())

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict[constant.UNK_WORD] = constant.UNK
        worddict[constant.BOS_WORD] = constant.BOS
        worddict[constant.EOS_WORD] = constant.EOS
        worddict[constant.PAD_WORD] = constant.PAD
        offset = len(worddict)
        idx = offset
        for ww in (sorted_words):
            if ww not in worddict :
                worddict[ww] = idx
                idx += 1

        wordfreq = OrderedDict()
        wordfreq[constant.UNK_WORD] = MAXVAL
        wordfreq[constant.BOS_WORD] = MAXVAL
        wordfreq[constant.EOS_WORD] = MAXVAL
        wordfreq[constant.PAD_WORD] = MAXVAL 
        for ww in (sorted_words) :
            if ww not in wordfreq :
                wordfreq[ww] = dict_freq[ww]

        with open('%s.dict'%filename, 'w') as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)
        with open('%s.freq'%filename, 'w') as f:
            json.dump(wordfreq, f, indent=2, ensure_ascii=False)
        print('Done')

if __name__ == '__main__':
    main()
