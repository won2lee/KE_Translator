#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    
    m_s = max([len(s) for s in sents])
    sents_padded = [[w for w in s] + [pad_token] * (m_s - len(s)) for s in sents]
    
    ### END YOUR CODE

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['_','<s>'] + sent + ['_','</s>']
        data.append(sent)

    return data

def get_sents_lenth4(source, sbol):

    if type(source[0]) is not list:
        source = [source]
    src_len = [len(s) for s in source]

    XO = [[(k,0) if i >0 and s[i-1] in sbol[0] else (k,1) for i,k in enumerate(s) if k!=sbol[1]] for j,s in enumerate(source) ]
    XO = [[(k,0) if k in sbol[0] or v==0 else (k,1) for (k,v) in s] for s in XO]
    XO = [[0]+[v for (k,v) in s][1:]+[0] for s in XO]

    XX = [[i for i,v in enumerate(s) if v==0] for s in XO]   
    XX = [[s[i]-s[i-1] for i in range(len(s)) if i>0] for s in XX]     # index to interval lenth(어절의 길이)
    XO = [s[:-1] for s in XO]
    return XX, XO  #, XK  # XX: Cutter, XO: lookup target list

def get_sents_lenth4_new(source, sbol):

    if type(source[0]) is not list:
        source = [source]
    src_len = [len(s) for s in source]     
    
    #   _<s>^  p  가  _ 계속 _ 오른 다 _  .  _ </s>
    #   1 0 2  0  0  1  0  1  0  0  1  0  1  0
    XO = [[sbol[k] if k in sbol.keys() else 0 for k in s] for s in source ]
    #   1 0 2  0  0  1  0  1  0  0  1  0  1  0       <= XO
    #   0   2        5     7        10   12   [14]   <= XX1
    #     1       4     6        9    11    13       <= XX_R
    #   2,  3,       2,    3,       2     2          <= XX      sum(XX) == len(s)
    #   1 0 1  0  0  1  0  1  0  0  1 0   1  0
    #   1   2  0     1     1  0     1     1          <= XO (0~3 사이의 값)
    #   1,  2,       1,    2,       1     1          <= X_sub   sum(X_sub) == len(XO)
    XX = [[i for i,v in enumerate(s) if v!=0]+[len(s)] for s in XO]    # XX1
    XX_R = [[k-1 for k in s[1:]] for s in XX]
    XX = [[s[i]-s[i-1] for i in range(len(s)) if i>0] for s in XX]     # index to interval lenth(어절의 길이)
    #XO = [ for i,k in enumerate(s) if k>0 or (k==0 and s[i+1] ==0]for s in XO]}
    XO = [[k for i,k in enumerate(s) if i not in XX_R[j]] for j,s in enumerate(XO)]
    X_sub = [[k-1 if k>0 else 0 for k in s ] for s in XX]

    return XX, XO, X_sub  #, XK  # XX: Cutter, XO: lookup target list
    
def get_sents_lenth3(source, sbol):

    if type(source[0]) is not list:
        source = [source]
    src_len = [len(s) for s in source]

    XO = [[(k,0) if i+1 < src_len[j] and s[i+1] in sbol[0] else (k,1) for i,k in enumerate(s) if k!=sbol[1]] for j,s in enumerate(source) ]
    XO = [[(k,0) if k in sbol[0] or v==0 else (k,1) for (k,v) in s] for s in XO]
    XO = [[v for (k,v) in s][:-1]+[0] for s in XO]

    XX = [[i for i,v in enumerate(s) if v==0] for s in XO]   
    XX = [[s[i]-s[i-1] if i>0 else s[i]+1 for i in range(len(s))] for s in XX]     # index to interval lenth(어절의 길이)

    return XX, XO  #, XK  # XX: Cutter, XO: lookup target list



def get_sents_lenth2(source, sbol):

    if type(source[0]) is not list:
        source = [source]
    
    source_lengths = [len(s) for s in source]
    XX = [list(chain(*[[i,i+1] for i,k in enumerate(s) if k in sbol[0]])) for s in source]
    to_add = [[i+1 for i,k in enumerate(s) if k == sbol[1]] for s in source]
    XX = [sorted(list(set(XX[i] +[source_lengths[i]]))) for i in range(len(XX))]   #len(XX): Batch size
    #to_sub = [[i for i,x in enumerate(xx) if x in to_add[j]] for j, xx in enumerate(XX)]
    XX = [[s[i]-s[i-1] if i>0 else s[i] for i in range(len(s))] for s in XX]     # index to interval lenth(어절의 길이)
    XX_len = [len(s) for s in XX]    # 문장의 어절 갯수
    #XX_subtracted = [[x-1 if i in to_sub[j] else x for i,x in enumerate(xx)] for j, xx in enumerate(XX)]
    return XX_len, XX  #, XX_subtracted


def get_sents_lenth(source, sbol):

    if type(source[0]) is not list:
        source = [source]
    
    source_lengths = [len(s) for s in source]
    XX = [list(chain(*[[i,i+1] for i,k in enumerate(s) if k in sbol[0]])) for s in source]
    to_add = [[i+1 for i,k in enumerate(s) if k == sbol[1]] for s in source]
    XX = [sorted(list(set(XX[i] + to_add[i]+[source_lengths[i]]))) for i in range(len(XX))]   #len(XX): Batch size
    to_sub = [[i for i,x in enumerate(xx) if x in to_add[j]] for j, xx in enumerate(XX)]
    XX = [[s[i]-s[i-1] if i>0 else s[i] for i in range(len(s))] for s in XX]     # index to interval lenth(어절의 길이)
    XX_len = [len(s) for s in XX]    # 문장의 어절 갯수
    XX_subtracted = [[x-1 if i in to_sub[j] else x for i,x in enumerate(xx)] for j, xx in enumerate(XX)]
    return XX_len, XX, XX_subtracted

def get_sents_lenth_new(source, sbol):

    if type(source[0]) is not list:
        source = [source]
    
    #    ^  p  가  _ 계속 _ 오른 다 _  .
    #    0         3     5       8    [10]       <=XX
    #    3         2     3       2               <=XX  XX_len = 4
    #    2         1     2       1               <=XX_subracted

    source_lengths = [len(s) for s in source]
    #XX = [list(chain(*[[i,i+1] for i,k in enumerate(s) if k in sbol[0]])) for s in source]
    XX = [[i for i,k in enumerate(s) if k in sbol] for s in source]
    XX = [XX[i] + [source_lengths[i]] for i in range(len(XX))]   #len(XX): Batch size
    #to_sub = [[i for i,x in enumerate(xx) if x in to_add[j]] for j, xx in enumerate(XX)]
    XX = [[s[i]-s[i-1] for i in range(len(s)) if i>0] for s in XX]     # index to interval lenth(어절의 길이)
    XX_len = [len(s) for s in XX]    # 문장의 어절 갯수
    XX_subtracted = [[x-1 if x>0 else 0 for x in xx ] for xx in XX]
    return XX_len, XX, XX_subtracted

def get_sent_lenth(s,sbol):
    to_add = [i+1 for i,k in enumerate(s) if k in sbol]
    """"
    X = list(chain(*[[i,i+1] for i,k in enumerate(s) if k in sbol[0]]))
    to_add = [i+1 for i,k in enumerate(s) if k in sbol]
    X = sorted(list(set(X + to_add+[len(s)])))
    #to_sub = [i for i,x in enumerate(X) if x in to_add] 
    X = [X[i]-X[i-1] if i>0 else X[i] for i in range(len(X))]
    #X = [x-1 if i in to_sub else x for i,x in enumerate(X)]
    """
    return len(s) - len(to_add)


def batch_iter(data, batch_size, slang, tlang, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    
    sbol = ['_','^','`']
    
    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        if slang == tlang:
            src_sents = []
            for e in examples:
                s = e[0]
                xnum1 = len(s)//4
                xnum2 = len(s)//10        
                xl = random.sample(s,xnum1)
                xnoise = ['<unk>']*(xnum1-xnum2)+random.sample(xl,xnum2)
                src_sents.append([[w if w not in xl else random.choice(xnoise) for w in s],e[1]])        
        
            examples = src_sents
        """
        if slang == tlang:
            src_sents = []
            for e in examples:
                s = e[0]
                sl = len(s)
                sent = []
                for i,w in enumerate(s):
                    nr = np.random.randint(10)
                    if nr < 1: 
                        sent.append('<unk>')
                    elif nr > 4:
                        sent.append(w)                   
                    else:
                        l = np.minimum(sl-i,4)
                        k = np.random.randint(l)
                        sent.append(s[i+k])
                        s[i+k] = w
                src_sents.append([sent,e[1]])
            examples = src_sents
        """
        if slang == '1':
            examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        else:
            examples = sorted(examples, key=lambda e: get_sent_lenth(e[0],sbol), reverse=True)
        
        """ 
        if slang == tlang:
            src_sents = []
            for e in examples:
                s = e[0]
                xnum1 = len(s)//4
                xnum2 = len(s)//10        
                xl = random.sample(s,xnum1)
                xnoise = ['<unk>']*(xnum1-xnum2)+random.sample(xl,xnum2)
                src_sents.append([w if w not in xl else random.choice(xnoise) for w in s])
        else:          
            src_sents = [e[0] for e in examples]
        """
        src_sents = [e[0] for e in examples]    
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
        