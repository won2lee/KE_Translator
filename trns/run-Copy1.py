#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --mono-en=<file> --mono-ko=<file> --vocab=<file> --map-en=<file> --map-ko=<file> --slang=<str> --tlang=<str> --mapping=<int>  --map_learning=<int> --back_trans=<int> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE --slang=<str> --tlang=<str> 
    run.py backtr [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE --slang=<str> --tlang=<str> --mapping=<int> --batch-size=<int>

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --mono_en=<file>                        mono english file
    --mono_ko=<file>                        mono korean file
    --vocab=<file>                          vocab file
    --map-en=<file>                         mapping en file
    --map-ko=<file>                         mapping ko file
    --slang=<str>                           source language 
    --tlang=<str>                           target language 
    --batch_ratio=<int>                     batch_ratio [default: 2]
    --loss_ratio=<int>                      loss_ratio [default: 2]
    --self_learning=<int>                   self_learning [default: 0]
    --mapping=<int>                         mapping [default: 0]
    --map_learning=<int>                    map_learning [default: 0]
    --back_trans=<int>                      back_trans [default: 0]
    --batch_plus=<int>                      batch_plus [default: 0]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 24]  
    --embed-size=<int>                      embedding size [default: 300]
    --hidden-size=<int>                     hidden size [default: 300]
    --clip-grad=<float>                     gradient clipping [default: 3.0]
    --log-every=<int>                       log every [default: 100]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 60]
    --max-num-trial=<int>                   terminate training after how many trials [default: 10]
    --lr-decay=<float>                      learning rate decay [default: 0.8]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.0010]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 400]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 100]
"""
import math
import sys
import pickle
import time


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nmt_model import Ehypothesis, Khypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter, sents_ordering
from vocab import Vocab, VocabEntry, generate_wid2cid, get_wid2cid
from update_mapping import net_run

import torch
import torch.nn.utils


def evaluate_ppl(model, dev_data, slang, tlang, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size, slang, tlang):
            loss = -model(src_sents, tgt_sents, slang=slang, tlang=tlang).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references, hypotheses): #: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    ##########################
    train_data_src = read_corpus(args['--train-tgt'], source='src')
    train_data_tgt = read_corpus(args['--train-src'], source='tgt')
    train_data_reverse = list(zip(train_data_src, train_data_tgt)) 

    train_data_src = read_corpus(args['--mono-en'], source='src')
    train_data_tgt = read_corpus(args['--mono-en'], source='tgt')
    train_data_eng = list(zip(train_data_src, train_data_tgt)) 

    train_data_src = read_corpus(args['--mono-ko'], source='src')
    train_data_tgt = read_corpus(args['--mono-ko'], source='tgt')
    train_data_kor = list(zip(train_data_src, train_data_tgt)) 

    #########################   

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'])
    #wid2cid, char_size = generate_wid2cid()
    #print("char_size : {}".format(char_size))
    wid2cid = get_wid2cid()
    char_size = 85
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                char_size = char_size,
                vocab=vocab,
                wid2cid = wid2cid)
    
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    #vocab_mask = torch.ones(len(vocab.tgt))
    #vocab_mask[vocab.tgt['<pad>']] = 0
    vocab_mask = torch.ones(len(vocab.vocs))
    vocab_mask[vocab.vocs['<pad>']] = 0
    
    #device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params['state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(torch.load(model_save_path + '.optim'))


    #[-8.638826, -6.454711, -5.433441, -4.800139, -4.392388, -4.252897,]
    #hist_valid_scores =  [-4.167480, -4.077592, -4.015549, -3.985279, -3.936693, -3.922876] #-4.019014, -3.957752]
    # [-8.619302, -6.616564, -5.923417, -5.611768, -5.393306, -5.183124, -5.146531]
    #hist_valid_scores =  [-4.941076, -4.779236, -4.738506, -4.714721, -4.6748542, -4.630830] 
    #hist_valid_scores =  [-8.414191, -6.489102, -5.838651, -5.50638, -5.317802, -5.157086, -5.105885,-4.824968, -4.743827]
    #hist_valid_scores = [-4.681234, -4.583382, -4.515263, -4.496487, -4.403906, -4.391593, -4.364640, -4.300215, -4.264980]
    #hist_valid_scores = [-4.250768, -4.230661]--map-en=./net_Module/mapping_en.pth --map-ko=./net_Module/mapping_ko.pth
    #hist_valid_scores = [-7.722062, -6.081345, -5.477855, -5.215838, -5.195213, -4.898636, -4.636147, -4.611800, -4.36, -4.229432]
    #hist_valid_scores = [-4.164422, -4.157305]
    #hist_valid_scores = [-4.266395]
    #hist_valid_scores = [-4.743406, -4.645375, -4.615770, -4.598774, -4.565829, -4.525113, -4.495907, -4.448615, -4.441942]
    #hist_valid_scores = [-4.432999, -4.419351, -4.387999, -4.382466] #to test the impact of changing loss weight
    #hist_valid_scores = [-6.695127, -6.303548, -5.595847, -5.252660, -5.048841, -4.891780, -4.762966, -4.688643, -4.641857, -4.561232] #bi
    #hist_valid_scores = [-4.449302, -4.386638, -4.311114, -4.264043] 
    #hist_valid_scores = [-4.405984, -4.394458, -4.331256, -4.252364, -4.216761, -4.188608, -4.172277, -4.150409, -4.119124]
    #hist_valid_scores = [-4.058765, -4.038687, -4.013042, -4.005287, -3.960983, -3.940266, -3.938432, -3.919321, -3.890531, -3.839670]
    #hist_valid_scores = [-3.829139, -3.802211, -3.751394, -3.736311, -3.723879]  # lr = 0.0001
    #hist_valid_scores = [-4.217068, -4.132213, -4.038739, -3.957581, -3.939603, -3.924544, -3.895795, -3.871743, -3.867299]
    #hist_valid_scores = [-3.857927, -3.842949, -3.826676, -3.816163, -3.784871, -3.775692, -3.716660, -3.708444, -3.706249, -3.701896] #-3.812397]
    #hist_valid_scores = [-3.706794, -3.702014, -3.667573, -3.654761, -3.657668] #[-3.692868, - 3.679998, -3.669266] #-3.687036, -3.681315]
    #hist_valid_scores = [-3.769498, -3.670924, -3.668278, -3.662438, -3.659656, -3.654382] #3.661110   
    #hist_valid_scores = [-5.506896, -5.255074, -5.092148, -4.825773, -4.743781, -4.608395, -4.556144, -4.496755]
    #hist_valid_scores = [-4.398007, -4.347569, -4.317553, -4.241752, -4.220772, -4.196398, -4.175547]
    hist_valid_scores = [-4.108687, -4.093388, -4.042322, -4.024141, -4.011558, -3.982819, -3.966806, -3.931323]
    ################   
    slang=args['--slang']
    tlang=args['--tlang']
    reverse_batch = batch_iter(train_data_reverse, train_batch_size, tlang, slang, shuffle=True) 

    b_ratio = int(args['--batch_ratio'])
    l_ratio = int(args['--loss_ratio'])
    self_learning_X = int(args['--self_learning'])
    mapping_X = int(args['--mapping'])
    map_learning = int(args['--map_learning'])
    back_trans = int(args['--back_trans'])
    batch_plus = int(args['--batch_plus'])


    mapping = 0

    """
    #bi_ratio self_ratio map_ratio
    if bitext_learning:
        enko_batch = batch_iter(train_data, train_batch_size//bi_ratio, slang, tlang, shuffle=True)
        koen_batch = batch_iter(train_data_reverse, train_batch_size//bi_ratio, tlang, slang, shuffle=True) 
        #eng_batch = batch_iter(train_data_eng, train_batch_size//b_ratio, slang, slang, shuffle=True) 
        #kor_batch = batch_iter(train_data_kor, train_batch_size//b_ratio, tlang, tlang, shuffle=True)      
    """
    if self_learning_X==1:
        eng_batch = batch_iter(train_data_eng, train_batch_size//b_ratio, slang, slang, shuffle=True) 
        kor_batch = batch_iter(train_data_kor, train_batch_size//b_ratio, tlang, tlang, shuffle=True)
   
    if mapping_X==1:
        #map_en = torch.load(args['--map-en']).to(device)
        #map_ko = torch.load(args['--map-ko']).to(device)
        eng_mapping_batch = batch_iter(train_data_eng, train_batch_size//b_ratio, slang, slang, shuffle=True, mapping=True) 
        kor_mapping_batch = batch_iter(train_data_kor, train_batch_size//b_ratio, tlang, tlang, shuffle=True, mapping=True)

    if back_trans==1:
        en_back_batch = batch_iter(train_data_eng, train_batch_size//b_ratio + batch_plus, slang, tlang, shuffle=True) 
        ko_back_batch = batch_iter(train_data_kor, train_batch_size//b_ratio + batch_plus, tlang, slang, shuffle=True)

    langs = [slang, tlang]

    #################  

    print("slang : {}, tlang : {}".format(slang, tlang))


    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, train_batch_size, slang, tlang, shuffle=True):
            train_iter += 1

            if self_learning_X==1: 
                self_learning=0 if train_iter % 1200 > 400 else 1
            else:
                self_learning =0

            if mapping_X==1:
                #mapping = 1
                mapping=0 if train_iter % 30000 > 4000 else 1
            else:
                mapping =0
            
            if back_trans==1 and train_iter % valid_niter == 1: 
                b_trns = {}
                b_gen = [en_back_batch, ko_back_batch]
                cutlines = []
                sbol = ['_','^','`']
                print("start back-translation ................")
                for ik,sl in enumerate(['ko2en', 'en2ko']):
                    lenR = 0.55 if ik==0 else 0.70
                    back_src_sents = []
                    for i in range(valid_niter+valid_niter//2):
                        back_src_sents += next(b_gen[ik])[0]
                    #back_src_sents = [s for s in back_src_sents if len(s) > 1]
                    nmted_sents, sent_scores = greedy_search(model, back_src_sents, batch_size=train_batch_size * 8,
                                max_decoding_time_step= 100, slang = langs[ik%2], tlang = langs[(ik+1)%2], mapping = 0)
                    #print( langs[ik%2], langs[(ik+1)%2])
                    #print("len(back_src_sents), len(nmted_sents) : {}, {}".format(len(back_src_sents), len(nmted_sents)))
                    #train_data_back = list(zip(nmted_sents, back_src_sents)) 
                    #train_data_back = list(zip(list(zip(nmted_sents, back_src_sents[:len(nmted_sents)])),sent_scores))
                    temp_set = list(zip([s.split(' ') for s in nmted_sents], back_src_sents, sent_scores))
                    # 번역된 문장의 길이가 원문에 비해 너무 짧지 않고 중복되는 단어가 적고(문장길이/set길이) 스코어가 높은 번역 
                    train_data_back = []
                    for bi in temp_set:
                        b0 = [w for w in bi[0] if w not in sbol]
                        if len(bi[0])/len(bi[1]) > lenR:
                            train_data_back.append([bi[0],bi[1],len(b0)/len(set(b0))-bi[2]])

                    #print("translated, src_sent, score : {} {} {}".format(
                    #    train_data_back[10][0].split(' ')[:20],train_data_back[10][1][:20],train_data_back[10][2]))
                    #print("len(train_data_back) : {} {}".format(len(train_data_back), train_data_back[-1])) 
                    cutline = sorted([bi[2] for bi in train_data_back])[valid_niter * (train_batch_size//b_ratio + batch_plus)]
                    cutlines.append(cutline)
                    train_data_back = [[bi[0],bi[1]] for bi in train_data_back if bi[2] < cutline] 
                    #print(train_data_back[:2])
                    b_trns[sl] = batch_iter(train_data_back, train_batch_size//b_ratio + batch_plus, langs[(ik+1)%2], langs[ik%2], shuffle=True, back_trans=1) 
                print("{} sents are back-translated .......    cutlines : {}  {}".format(len(nmted_sents)*2,cutlines[0], cutlines[1]))

            optimizer.zero_grad()

            batch_size = len(src_sents)

            batch_loss = 0
            loss_base = 0
            #langs = [slang, tlang]

            ####################
            #if True: #mapping == 0:
            if mapping == 0 and self_learning == 0 and back_trans==0:
                example_losses1 = -model(src_sents, tgt_sents, slang, tlang) # (batch_size,)           
                src_sents2, tgt_sents2 = next(reverse_batch)  
                example_losses2 = -model(src_sents2, tgt_sents2, tlang, slang) # (batch_size,)
                batch_loss += example_losses1.sum() + example_losses2.sum()
                loss_base += 2
            #####################
            if self_learning==1:
                src_sents3, tgt_sents3 = next(eng_batch)  
                example_losses3 = -model(src_sents3, tgt_sents3, slang, slang) # (batch_size,)
                src_sents4, tgt_sents4 = next(kor_batch)  
                example_losses4 = -model(src_sents4, tgt_sents4, tlang, tlang) # (batch_size,)
                batch_loss += (example_losses3.sum() + example_losses4.sum()) * (1.0 / l_ratio)
                loss_base += 2 / (b_ratio * l_ratio)

            if back_trans==1:
                """
                
                for ik, lkey in enumerate(['en2ko','ko2en']):                   
                    if ik==1:
                        src_sents, tgt_sents = next(reverse_batch)
                    src_sents3, tgt_sents3 = next(b_trns[lkey]) 
                    src_sents = src_sents3
                    tgt_sents = src_sents3
                    #src_sents, tgt_sents = sents_ordering(list(zip(src_sents,tgt_sents)))
                    if ik == 0:
                        batch_loss += -model(src_sents, tgt_sents, slang, tlang).sum()
                    else:
                        batch_loss += -model(src_sents, tgt_sents, tlang, slang).sum()

                    #batch_loss += -model(src_sents, tgt_sents, langs[ik%2], langs[(ik+1)%2]).sum()
                loss_base += 2

                """
                src_sents2, tgt_sents2 = next(reverse_batch)

                src_sents3, tgt_sents3 = next(b_trns['en2ko'])
                #print('\n',src_sents3[7]) 
                #print(tgt_sents3[7],'\n')
                src_sents, tgt_sents = sents_ordering(list(zip(src_sents+src_sents3,tgt_sents+tgt_sents3)))
                example_losses3 = -model(src_sents, tgt_sents, slang, tlang) # (batch_size,)

                src_sents4, tgt_sents4 = next(b_trns['ko2en']) 
                src_sents2, tgt_sents2 = sents_ordering(list(zip(src_sents2+src_sents4,tgt_sents2+tgt_sents4))) 
                example_losses4 = -model(src_sents2, tgt_sents2, tlang, slang) # (batch_size,)

                batch_loss += (example_losses3.sum() + example_losses4.sum()) * (1.0 / l_ratio)
                loss_base += 2 #/ (b_ratio * l_ratio) 


                """
                src_sents3, tgt_sents3 = next(b_trns['en2ko'])
                #print('\n',src_sents3[7]) 
                #print(tgt_sents3[7],'\n')
                example_losses3 = -model(src_sents3, tgt_sents3, slang, tlang) # (batch_size,)
                src_sents4, tgt_sents4 = next(b_trns['ko2en'])  
                example_losses4 = -model(src_sents4, tgt_sents4, tlang, slang) # (batch_size,)
                batch_loss += (example_losses3.sum() + example_losses4.sum()) * (1.0 / l_ratio)
                loss_base += 2 / (b_ratio * l_ratio) 
                """     

            if mapping==1:   
                if train_iter == 1 or train_iter % 100 == 1:
                    trained_W = [model.map_en, model.map_ko]
                    nul_mapping = True if train_iter==1 else False
                    update_map = False

                    if train_iter == 1:
                        update_map = True
                       
                    elif map_learning==1:
                        m_diff = torch.norm(torch.mm(trained_W[0].weight.data,trained_W[1].weight.data) - torch.eye(300).to(device))
                        if m_diff > 1.0:   
                            update_map = True                     
                            print("torch.mm(W[0],W[1] is not Identity Matrix. difference = {}".format(m_diff))
                    elif train_iter % 400 == 1:
                        update_map = True
                    
                    if update_map==1:
                        with torch.no_grad():
                            map_en, map_ko,dict_mapping = net_run(model.model_embeddings, model.vocab, trained_W, nul_mapping, device) 
                        model.map_en.weight.data = map_en if train_iter == 1 else 0.4 * map_en + 0.6 * model.map_en.weight.data
                        model.map_ko.weight.data = map_ko if train_iter == 1 else 0.4 * map_ko + 0.6 * model.map_ko.weight.data 
                        model.dict_mapping = dict_mapping

                src_sents5, tgt_sents5 = next(eng_mapping_batch)  
                example_losses5 = -model(src_sents5, tgt_sents5, slang, slang, mapping=True) # (batch_size,)
                src_sents6, tgt_sents6 = next(kor_mapping_batch)  
                example_losses6 = -model(src_sents6, tgt_sents6, tlang, tlang, mapping=True) # (batch_size,)
                batch_loss += (example_losses5.sum() + example_losses6.sum()) * (1.0 / l_ratio)
                loss_base += 2 / (b_ratio * l_ratio)      
            #example_losses = -model(src_sents, tgt_sents, slang, tlang) # (batch_size,)
            #batch_loss = example_losses.sum() 
            batch_loss = batch_loss / loss_base
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                #print("learning rate : {}".format(lr))

                # compute dev. ppl and bleu, dev batch size can be a bit larger
                dev_ppl = evaluate_ppl(model, dev_data, slang, tlang, batch_size=32)   
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
                
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores) - (patience + 1) * (0.0002 +mapping*0.01)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0
                """
                if mapping:
                    with torch.no_grad():
                        map_en, map_ko, dict_mapping = net_run(model.model_embeddings, model.vocab, device)
                    model.map_en.weight.data = 0.4 * map_en + 0.6 * model.map_en.weight.data
                    model.map_ko.weight.data = 0.4 * map_ko + 0.6 * model.map_ko.weight.data
                    model.dict_mapping = dict_mapping
                """
                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    wid2cid = get_wid2cid()
    char_size = 85
    model = NMT.load(args['MODEL_PATH'], char_size=char_size, wid2cid=wid2cid)

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']),
                             tlang = args['--tlang'])

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')

def beam_search(model, test_data_src, beam_size, max_decoding_time_step, tlang): 
    """
    def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) 
    -> List[List[Hypothesis]]:
    Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    print(tlang, "tlang == ko : {}".format(tlang == 'ko' ))

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.ek_beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step,tlang=tlang)
            """
            if tlang =='en':
                example_hyps = model.ke_beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            else:
                example_hyps = model.ek_beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            """
            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def decode_greedy(args: Dict[str, str]):
 
    print("load sentences for back_Trans from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)

    wid2cid = get_wid2cid()
    char_size = 85
    model = NMT.load(args['MODEL_PATH'], char_size=char_size, wid2cid=wid2cid)
    #model = NMT.load(args['MODEL_PATH'])
    

    mapping = args['--mapping']
    #print(mapping)
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    model.to(device)

    if mapping==1:   
        with torch.no_grad():
            map_en, map_ko,dict_mapping = net_run(model.model_embeddings, model.vocab, device) 
        model.map_en.weight.data = map_en 
        model.map_ko.weight.data = map_ko 
        model.dict_mapping = dict_mapping

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    nmted_sents, sent_scores = greedy_search(model, test_data_src,
                             batch_size=int(args['--batch-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']),
                             slang = args['--slang'],
                             tlang = args['--tlang'],
                             mapping = 0) #mapping)
    """
    if args['TEST_TARGET_FILE']:
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, nmted_sents)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)
    """
    to_add = '_mapping' if mapping==1 else ''
    print("length of nmted : {}".format(len(nmted_sents)))

    with open(args['OUTPUT_FILE']+to_add, 'w') as f:
        for src_sent, nmted, score in list(zip(test_data_src[:len(nmted_sents)], nmted_sents, sent_scores)):
            sent = ''.join(nmted)
            src = ' '.join(src_sent)
            #with open('outputs/bcktr_en2ko.txt', 'w' if i == 1 else 'a') as f:
            f.write(str(score) + '\n' + src +'\n' + sent + '\n' +'\n')

def greedy_search(model, test_data_src, batch_size, max_decoding_time_step, slang, tlang, mapping): 

    was_training = model.training
    model.eval()

    #print(tlang, "tlang == ko : {}, mapping : {}".format(tlang == 'ko', mapping ))

    nmted_sents = []
    sents_scores = []
    n_iter = len(test_data_src) // batch_size + 1
    with torch.no_grad():
        #for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        for ik in range(n_iter):
            src_snts = test_data_src[ik*batch_size:(ik+1)*batch_size]
            if len(src_snts) <2:continue
            sents, scores = model.greedy_search(src_snts, max_decoding_time_step=max_decoding_time_step,slang=slang,tlang=tlang,mapping=mapping)
            #print("length of sents : {}, sents[0][:3] : {}".format(len(sents), sents[0][:3]))
            nmted_sents += sents 
            sents_scores += scores
        #print("length of nmted_sents : {}, sents[0][:3] : {}".format(len(nmted_sents), nmted_sents[10][:100]))
        #print(src_snts[10][:100])
    if was_training: model.train(was_training)

    return nmted_sents, sents_scores


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    elif args['backtr']:
        decode_greedy(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
