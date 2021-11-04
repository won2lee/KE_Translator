# trns_koren.py

import os
import re
import torch
import torch.nn.utils
#from flaskr.vocab import Vocab, VocabEntry


#from flaskr.nmt_model import Hypothesis, NMT

class Trns(object):
    
    def __init__(self, model):
        
        self.model = model  #NMT.load('/home/john/flaskr/model.bin')
        
                
    def translate(self, test_data_src, tlang):
        """ Performs decoding on a test set, and save the best-scoring decoding results.
        If the target gold-standard sentences are given, the function also computes
        corpus-level BLEU score.
        @param args (Dict): args from cmd line
        """

        hypotheses = self.beam_search(self.model, test_data_src,
                                 beam_size=int(10),
                                 max_decoding_time_step=int(200),
                                 tlang = tlang)

        sents = []

        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value) 
            sents.append(hyp_sent)

        return sents

    def beam_search(self, model, test_data_src, beam_size, max_decoding_time_step, tlang):
        """ Run beam search to construct hypotheses for a list of src-language sentences.
        @param model (NMT): NMT Model
        @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
        @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
        @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
        @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
        """
        #was_training = model.training
        model.eval()

        hypotheses = []
        with torch.no_grad():
            for src_sent in test_data_src:

                example_hyps = model.ek_beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, tlang=tlang)

                hypotheses.append(example_hyps)

        #if was_training: model.train(was_training)

        return hypotheses
    
