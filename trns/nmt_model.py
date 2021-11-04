#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from trns.utils import get_sents_lenth4_new, get_sents_lenth_new, get_notEn, get_X_cap
from itertools import chain

from trns.model_embeddings import ModelEmbeddings
from trns.char_embeddings import CharEmbeddings

Ehypothesis = namedtuple('Ehypothesis', ['value', 'score'])
Khypothesis = namedtuple('Khypothesis', ['value', 'xo', 'score', 'u_score', 'a_score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, char_size, vocab, wid2cid, dropout_rate=0.2): #char_size
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.char_embeddings = CharEmbeddings(char_size, embed_size, hidden_size)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.wid2cid = wid2cid
        #self.map_en = map_en.to("cuda") #torch.tensor(map_en, dtype=torch.float, device=self.device)
        #self.map_ko = map_ko.to("cuda") #torch.tensor(map_ko, dtype=torch.float, device=self.device)  

        self.token = ['(', ')', ',', "'", '"','_','<s>','</s>']
        self.sbol = ['_','^','`']
        self.ko_start = 54621
        self.xo_weight = 1.0
        self.notEn = get_notEn(self.vocab)
        #self.sbol_padded = self.vocab.vocs.to_input_tensor([['_'],['^'],['`']], device=self.device)

        # default values

        self.map_en = None 
        self.map_ko = None

        self.en_encoder = None 
        self.ko_decoder = None
        self.en_h_projection = None
        self.en_c_projection = None
        self.ko_att_projection = None
        self.ko_combined_output_projection = None
        #self.ek_target_vocab_projection = None
        
        self.ko_encoder = None 
        self.en_decoder = None
        self.ko_h_projection = None
        self.ko_c_projection = None
        self.en_att_projection = None
        self.en_combined_output_projection = None
        #self.ke_target_vocab_projection = None 

        self.target_vocab_projection = None   
        
        self.dropout = None
        self.dropout_10 = None

        self.sub_encoder= None
        self.en_gate = None
        self.sub_en_projection = None  
        
        self.sub_decoder= None
        self.de_gate = None
        self.sub_de_projection = None
        
        self.target_ox_projection = None

        ### Initialize the following variables:
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.decoder (LSTM Cell with bias)
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        ###     self.dropout (Dropout Layer)
        ###
        ### Use the following docs to properly initialize these variables:
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout

        self.en_encoder = nn.LSTM(embed_size, self.hidden_size, num_layers=2, bidirectional=True)
        self.ko_decoder = nn.LSTM(embed_size+self.hidden_size, self.hidden_size, num_layers=2)
        self.en_h1_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.en_h2_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)   
        self.en_c1_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.en_c2_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.ko_att_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.ko_combined_output_projection = nn.Linear(3*self.hidden_size, self.hidden_size, bias=False)
        #self.ek_target_vocab_projection = nn.Linear(self.hidden_size, len(self.vocab.vocs), bias=False)        
                
        self.ko_encoder = nn.LSTM(embed_size, self.hidden_size, num_layers=2, bidirectional=True)
        self.en_decoder = nn.LSTM(embed_size+self.hidden_size, self.hidden_size, num_layers=2)
        self.ko_h1_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.ko_h2_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)   
        self.ko_c1_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.ko_c2_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.en_att_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.en_combined_output_projection = nn.Linear(3*self.hidden_size, self.hidden_size, bias=False)
        #self.ke_target_vocab_projection = nn.Linear(self.hidden_size, len(self.vocab.vocs), bias=False)

        self.target_vocab_projection = nn.Linear(self.hidden_size, len(self.vocab.vocs), bias=False)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.dropout_10 = nn.Dropout(p=0.3)
        """       
        self.sub_encoder= nn.LSTM(embed_size, self.hidden_size, bidirectional=True)
        self.en_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sub_en_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)  
        
        self.sub_decoder= nn.LSTM(embed_size, self.hidden_size)
        self.de_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sub_de_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False) 
        """
        self.sub_en_coder= nn.LSTM(embed_size, self.hidden_size)
        self.en_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sub_en_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False) 

        self.sub_ko_coder= nn.LSTM(embed_size, self.hidden_size)
        self.ko_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sub_ko_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False) 
        
        self.cap_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    
        self.target_ox_projection = nn.Linear(self.hidden_size, 4, bias=False)  

        self.map_en = nn.Linear(embed_size, embed_size, bias=False) #,requires_grad = False) 
        self.map_ko = nn.Linear(embed_size, embed_size, bias=False) #,requires_grad = False)
        #self.map_en.weight.data = map_en #torch.tensor(map_en,device=self.device) 
        #self.map_ko.weight.data = map_en #torch.tensor(map_en,device=self.device)
        self.map_en.weight.requires_grad = False
        self.map_ko.weight.requires_grad = False
        self.count = 0   
        self.pre_map = None 
        self.dict_mapping = {}
        self.mask_X_ratio  = 0.5
        self.X_count = 0

        self.en_projs = [self.en_h1_projection, self.en_h2_projection,self.en_c1_projection,self.en_c2_projection, self.ko_h1_projection, self.ko_h2_projection,self.ko_c1_projection,self.ko_c2_projection]       
        self.de_projs = [self.target_vocab_projection, self.en_att_projection, self.ko_att_projection, self.en_combined_output_projection, self.ko_combined_output_projection]
        self.mapping_grad = 1 #True
        self.st_grad = 1
        self.bt_grad = 1

        """
        self.ko_decoder.requires_grad = False
        self.en_decoder.requires_grad = False
        self.en_att_projection.weight.requires_grad = False
        self.ko_att_projection.weight.requires_grad = False       
        self.en_combined_output_projection.weight.requires_grad = False
        self.ko_combined_output_projection.weight.requires_grad = False
        self.target_vocab_projection.weight.requires_grad = False
        """

    def forward(self, source: List[List[str]], target: List[List[str]], slang, tlang, mapping=0, self_trns=0, back_trns=0): # -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each ex0ample in the input batch. Here b = batch size.
        """
        
        #################################################################### 
        
        if  mapping==1 and self.mapping_grad==1: #back_trns ==1 or  mapping==1:
            self.freeze_params(self.ko_encoder, unfreeze =False)
            self.freeze_params(self.en_encoder, unfreeze =False)
            for proj in  self.en_projs:
                self.freeze_params(proj, unfreeze =False)
            self.mapping_grad=0     #requires_grad : False           
        
        elif mapping==0 and self.mapping_grad==0:
            self.freeze_params(self.ko_encoder, unfreeze =True)
            self.freeze_params(self.en_encoder, unfreeze =True) 
            for proj in  self.en_projs:
                self.freeze_params(proj, unfreeze =True) 
            self.mapping_grad=1     #requires_grad : True ===> 계속 중복 지정하지 않도록 

        #"""         
        if self_trns ==1 and self.st_grad==1:
            self.freeze_params(self.ko_decoder, unfreeze =False)
            self.freeze_params(self.en_decoder, unfreeze =False)
            self.freeze_params(self.target_vocab_projection, unfreeze =False)
            for proj in  self.de_projs:
                self.freeze_params(proj, unfreeze =False)
            self.st_grad=0

        elif self_trns ==0 and self.st_grad==0:
            self.freeze_params(self.ko_decoder, unfreeze =True)
            self.freeze_params(self.en_decoder, unfreeze =True)
            self.freeze_params(self.target_vocab_projection, unfreeze =True)
            for proj in  self.de_projs:
                self.freeze_params(proj, unfreeze =True)
            self.st_grad=1
        #"""
        
        if  back_trns == 1 and self.bt_grad==1: #back_trns ==1 or  mapping==1:
            self.freeze_params(self.ko_encoder, unfreeze =False)
            self.freeze_params(self.en_encoder, unfreeze =False)
            for proj in  self.en_projs:
                self.freeze_params(proj, unfreeze =False)
            self.bt_grad=0     #requires_grad : False
        
        elif back_trns == 0 and self.bt_grad==0: 
            self.freeze_params(self.ko_encoder, unfreeze =True)
            self.freeze_params(self.en_encoder, unfreeze =True) 
            for proj in  self.en_projs:
                self.freeze_params(proj, unfreeze =True) 
            self.bt_grad =1     #requires_grad : True ===> 계속 중복 지정하지 않도록 

        if mapping==1:            
            #if self.count % 10 == 0: #slang == tlang: 
            #self.map_en.weight.data = map_en #torch.tensor(map_en,device=self.device) 
            #self.map_ko.weight.data = map_en #torch.tensor(map_en,device=self.device)

            """   
            xW = self.map_en.weight.data
            self.map_en.weight.data = 1.01 * xW - 0.01 * torch.mm(torch.mm(xW, xW.transpose(1,0)),xW)
            xW = self.map_ko.weight.data
            self.map_ko.weight.data = 1.01 * xW - 0.01 * torch.mm(torch.mm(xW, xW.transpose(1,0)),xW)
                         
            if self.count % 400 ==0:
                #print(self.count)
                if self.pre_map is not None:
                    moved_map = torch.norm(self.map_en.weight.data) # - self.pre_map)
                    #if moved_map >0.00000001:
                    print("map weight is moved !!! {} moved".format(moved_map))
                    #else:
                        #print("map weight is freezed !!! {} moved".format(moved_map)
                self.pre_map = self.map_en.weight.data
            self.count +=1
            """
            source_padded, source_lengths = self.parallel_encode_new(source, slang, mapping=1 ) #slang_is_tlang=True)
            slang = 'ko' if slang =='en' else 'en'
        else:
            source_padded, source_lengths = self.parallel_encode_new(source, slang)

        # enc_hiddens, dec_init_state = self.en_encode(source_padded, source_lengths) 
        #enc_hiddens, dec_init_state = self.en_encode(source_padded, source_lengths) 
        
        if slang == 'en':         
            enc_hiddens, dec_init_state = self.en_encode(source_padded, source_lengths, self_trns=self_trns)             
        else:
            enc_hiddens, dec_init_state = self.ko_encode(source_padded, source_lengths, self_trns=self_trns)   
        """
        source_lengths = [len(s) for s in source]
        # Convert list of lists into tensors
        source_padded = self.vocab.vocs.to_input_tensor(source, device=self.device) 
        source_embedded = self.model_embeddings.vocabs(source_padded) 
        enc_hiddens, dec_init_state = self.en_encode(source_embedded, source_lengths)
        """           
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
                               
        tgt = [[w for w in s if w not in self.sbol] for s in target]
        #tgt = target
        parallel_padded = self.vocab.vocs.to_input_tensor(target, device=self.device)  #Tensor: (tgt_len, b)
        target_padded = self.vocab.vocs.to_input_tensor(tgt, device=self.device) 
        target_embedded, XO = self.parallel_decode_new(target, parallel_padded, lang=tlang)    #parallel_padded[:-1]) 

        if tlang == 'ko':  
            combined_outputs = self.ko_decode(enc_hiddens, enc_masks, dec_init_state, target_embedded)             
        else:
            combined_outputs = self.en_decode(enc_hiddens, enc_masks, dec_init_state, target_embedded) 

        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)
        target_masks = (target_padded != self.vocab.vocs['<pad>']).float()
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]              
        P2 = F.log_softmax(self.target_ox_projection(combined_outputs), dim=-1)
        target_ox_log_prob = torch.gather(P2, index=XO[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0) + self.xo_weight * target_ox_log_prob.sum(dim=0)

        """    
        else:
            target_padded = self.vocab.vocs.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)
            combined_outputs = self.ke_decode(enc_hiddens, enc_masks, dec_init_state, target_padded) 
            P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)
            target_masks = (target_padded != self.vocab.vocs['<pad>']).float()
            target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:] 
            scores = target_gold_words_log_prob.sum(dim=0)
        """    
        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by theX `self.decode()` function.

        # Zero out, probabilities for which we have nothing in the target text
        #target_masks = (target_padded != self.vocab.vocs['<pad>']).float()
        
        # Compute log probability of generating true target words
        #target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
           
        return scores

    def freeze_params(self, nnF, unfreeze =False):           
        for para in nnF.parameters():
            para.requires_grad=unfreeze

    def parallel_encode(self,source):

        if type(source[0]) is not list:
           source = [source]       
    
        sbol = [['(', ')', ',', "'", '"'],'_']
        source_lengths, Z, Z_sub = get_sents_lenth(source,sbol) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in source]  # 원래의 문장 길이
 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        
        max_l = max(s_len)           
        XX =  [s+[max_l-s_len[i]] if max_l>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        src_padded = self.vocab.vocs.to_input_tensor(source, device=self.device)  
 
        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(
            torch.split(src_padded,1,-1))]))     #각 문장으로 자른 뒤 문장내 어절 단위로 자른다 

        Z_flat = list(chain(*Z_sub))
        X = [s[:Z_flat[i]]for i,s in enumerate(X)]
        X = pad_sequence(X).squeeze(-1)
        #X = torch.tensor(X, dtype=torch.float, device = self.device)

        X_embed = self.model_embeddings.vocabs(X)
        
        out,(last_h1,last_c1) = self.sub_encoder(X_embed)
        X_proj = self.sub_en_projection(out)
        X_gate = torch.sigmoid(self.en_gate(X_embed))
        X_way = self.dropout(X_gate * X_embed + (1-X_gate) * X_proj)       

        #문장단위로 자르고 어절 단위로 자른 뒤 각 어절의 길이만 남기고 나머지는 버린 후 연결 (cat) 하여 문장으로 재구성         
        X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
          torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        
        # 재구성된 문장의 길이가 다르기 때문에 패딩
        source_padded = pad_sequence(X_input).squeeze(-2)
        source_lengths = [sum([wl for wl in s]) for s in Z_sub]
        
        return source_padded, source_lengths

    def parallel_encode_new(self,source, lang, mapping=0): #slang_is_tlang=False):

        if type(source[0]) is not list:
           source = [source]       
    
        source_lengths, Z, Z_sub = get_sents_lenth_new(source,self.sbol) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in source]  # 원래의 문장 길이
 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        
        max_l = max(s_len)           
        XX =  [s+[max_l-s_len[i]] if max_l>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        src_padded = self.vocab.vocs.to_input_tensor(source, device=self.device)  
        """
        for i in range(len(source)):
            if len(src_padded[:,i]) != sum(XX[i]):
                print("s_len : {}, padded : {}, sumXX : {}".format(s_len[i],len(src_padded[:,i]), sum(XX[i])))
                print("XX : {}".format(XX[i]))
                print(source[i])
        """
        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(
            torch.split(src_padded,1,-1))]))     #각 문장(batch)으로 자른 뒤 문장내 어절 단위로 자른다 

        #Z_flat = list(chain(*Z_sub))
        #X = [s[:Z_flat[i]]for i,s in enumerate(X)]
        X = pad_sequence(X).squeeze(-1)
        #X = torch.tensor(X, dtype=torch.float, device = self.device)
        #X_embed = self.model_embeddings.vocabs(X)
        #print(X_embed.size(),self.map_en.size())
        
        if lang =='en':
            cap_id, len_X = get_X_cap(source, self.sbol)
            
        if mapping==1:  #slang_is_tlang:
             
            ##### if not map_learning
            """ 
            X_embed = self.model_embeddings.vocabs(torch.tensor([[self.dict_mapping[w.item()] for w in s] for s in X],device=self.device))
                     
            mask_X = torch.tensor([[1 if w.item()==0 or self.dict_mapping[w.item()]>0 else 0 for w in s] for s in X]).float().to(self.device)
            self.X_count += 1
            self.mask_X_ratio = (mask_X.sum() / (mask_X.size(0) * mask_X.size(1) * self.X_count) 
                                    + (self.X_count - 1) * self.mask_X_ratio / self.X_count)
            if self.X_count % 2000 == 1:
                print("self.X_count : {},  self.mask_X_ratio : {}".format(self.X_count, self.mask_X_ratio) )    
            #####
                            
            mask_X = mask_X.unsqueeze(-1)
            if lang == 'en':
                X_embed = X_embed * mask_X + self.map_en(X_embed * (1.0-mask_X))
            else:
                X_embed = X_embed * mask_X + self.map_ko(X_embed * (1.0-mask_X))

            if lang=='en':
                # 영어가 아니면 1, 영어이면  0
                mask_X = torch.tensor([[self.notEn[w.item()] for w in s] for s in X]).float().to(self.device) #notEn : not eng alphabet           
                # 번역된 단어로 바뀐 경우 0+, 번역되지 않은 경우 0
                dict_mask = torch.tensor([[abs(self.dict_mapping[w.item()]-w) for w in s] for s in X]).float().to(self.device)
                # masking : mapping 하지 않을 단어 1 (X_embed 그대로 사용), mapping 할 단어  0
                mask_X = (mask_X + dict_mask > 0).float().unsqueeze(-1)
                alpha = torch.randn_like(X_embed) * 0.2
                # mapping 에 노이즈 추가 
                #masked_X = X_embed * (1.0-mask_X)
                X_embed = X_embed * mask_X + self.map_en(X_embed * (1.0-mask_X)) * (1.+alpha) #+ masked_X * alpha
                
            else:         
                mask_X = (X < self.ko_start).float().to(self.device)  #한글이 아닌 경우
                dict_mask = torch.tensor([[abs(self.dict_mapping[w.item()]-w) for w in s] for s in X]).float().to(self.device)
                mask_X = (mask_X + dict_mask > 0.).float().unsqueeze(-1)
                alpha = torch.randn_like(X_embed) * 0.2
                #masked_X = X_embed * (1.0-mask_X)
                X_embed = X_embed * mask_X + self.map_ko(X_embed * (1.0-mask_X)) * (1.+alpha) #+ masked_X * alpha
            """
            ##### if map_learning
            
            k = torch.randint(5,(1,)) #dict_mapping 중 20%는 dict lookup 하지 않고 mapping
            if lang == 'en':  
                X_dict = torch.tensor([[-w.item() if (ik in cap_id) or 
                            (i%5!=k and self.dict_mapping[w.item()]>0 and self.dict_mapping[w.item()]!= w.item()) 
                        else self.dict_mapping[w.item()] for i,w in enumerate(s)] for ik,s in enumerate(X)],device=self.device)
            else:
                X_dict = torch.tensor([[-w.item() if (i%5!=k and self.dict_mapping[w.item()]>0 and self.dict_mapping[w.item()]!= w.item())
                        else self.dict_mapping[w.item()] for i,w in enumerate(s)] for ik,s in enumerate(X)],device=self.device)
                
            #X_dict = torch.tensor([[self.dict_mapping[w.item()] for i,w in enumerate(s)] for s in X],device=self.device)
            X_embed = self.model_embeddings.vocabs(torch.abs(X_dict))
            
        else:
            X_embed = self.model_embeddings.vocabs(X)
            
        if lang =='en':
            #cap_id, len_X = get_X_cap(source, self.sbol)
            if len(cap_id) >0: #beam search의 경우(sents 수가 작은 경우) 대문자 시작이 없는 경우 발생                 
                cap_ids = torch.tensor(cap_id, device=self.device) #get_X_cap(source, sbol))
                #
                """ comment => noChar version, activate => Char version
                #print("cap_ids.size(), len_X, X.size : {} {} {}".format(cap_ids.size(), len_X, torch.tensor(X).size))
                X_cap = torch.tensor([[self.wid2cid[s[0].item()] for s in sss] for sss in torch.split(X[:,cap_ids],1,-1)], device=self.device)
                cap_gate = torch.sigmoid(self.cap_gate(X_embed[:,cap_ids]))
                X_embed[:,cap_id] = cap_gate * X_embed[:, cap_id] + (1-cap_gate) * self.char_embeddings(X_cap).transpose(1,0)
                """
        if mapping==1:   
            
            mask_X = (X_dict > -1).float().to(self.device).unsqueeze(-1)
            if lang == 'en':
                X_embed = X_embed * mask_X + self.map_en(X_embed * (1.0-mask_X)) #* (1.+alpha) #+ masked_X * alpha
            else:
                X_embed = X_embed * mask_X + self.map_ko(X_embed * (1.0-mask_X))
            
            lang = 'ko' if lang=='en' else 'en'  # mapping을 거치면서 언어가 바뀐 것으로 취급  
            
  
        if lang == 'en':
            out,(last_h1,last_c1) = self.sub_en_coder(X_embed)
            X_proj = self.sub_en_projection(out[1:])               #sbol 부분 제거
            X_gate = torch.sigmoid(self.en_gate(X_embed[1:]))

        else:
            out,(last_h1,last_c1) = self.sub_ko_coder(X_embed)
            X_proj = self.sub_ko_projection(out[1:])               #sbol 부분 제거
            X_gate = torch.sigmoid(self.ko_gate(X_embed[1:]))
        #    #print("gate embed proj : {} {} {}".format(X_gate.size(), X_embed.size(), X_proj.size()))

        X_way = self.dropout(X_gate * X_embed[1:] + (1-X_gate) * X_proj)       

        #문장단위로 자르고 어절 단위로 자른 뒤 각 어절의 길이만 남기고 나머지는 버린 후 연결 (cat) 하여 문장으로 재구성         
        X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
          torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        
        # 재구성된 문장의 길이가 다르기 때문에 패딩
        source_padded = pad_sequence(X_input).squeeze(-2)
        source_lengths = [sum([wl for wl in s]) for s in Z_sub]
        
        return source_padded, source_lengths

    def parallel_decode_old(self,target, tgt_padded):
    
        sbol = [['(', ')', ',', "'", '"'],'_']
        if type(target[0]) is not list: target = [target]
        target_lengths, Z, Z_sub = get_sents_lenth(target,sbol) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in target]  # 원래의 문장 길이
 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        
        max_l = max(s_len)  
        Z = [s if max_l>s_len[i] else s[:-1]+[s[-1]-1] for i,s in enumerate(Z)]         
        XX =  [s+[max_l-s_len[i]-1] if max_l-1>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        #tgt_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  
 
        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(torch.split(tgt_padded,1,-1))]))

        #Z_flat = list(chain(*Z_sub))
        #X = [s[:Z_flat[i]]for i,s in enumerate(X)]

        X = pad_sequence(X).squeeze(-1)
        X = torch.tensor(X,device = self.device)
 
        X_embed = self.model_embeddings.target(X)
 
        X_gate = torch.sigmoid(self.gate(X_embed))
        out,(last_h1,last_c1) = self.sub_decoder(X_embed)
        X_proj = self.dropout(torch.tanh(self.sub_de_projection(out)))
        X_way = X_gate * X_embed + (1-X_gate) * X_proj       
        
        #X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
        #  torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        X_input = [torch.cat([ss[:Z[i][j]]for j,ss in enumerate(
            torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
          
        target_padded = pad_sequence(X_input).squeeze(-2)
        #target_lengths = [sum([wl for wl in s]) for s in Z_sub]
        
        return target_padded


    def parallel_decode(self,target, tgt,tgt_padded):
    
        sbol = [['(', ')', ',', "'", '"','_'],'_']
        if type(target[0]) is not list: target = [target]
        Z, XO = get_sents_lenth4(target,sbol) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in tgt]
        max_l = max(s_len)  
        Z = [s if max_l>s_len[i] else s[:-1]+[s[-1]-1] for i,s in enumerate(Z)] 
        Z = [[sx for sx in s if sx>0] for s in Z] 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        max_l = max_l - 1
        XX =  [s+[max_l-s_len[i]] if max_l>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        #tgt_padded = self.vocab.tgt.to_input_tensor(target, device=self.device) 
        """
        print(Z_len, tgt_padded.shape)
        for i in range(4):
            print(XX[i], sum(XX[i]))
            print(XO[i])
        """
 
        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(torch.split(tgt_padded,1,-1))]))

        #Z_flat = list(chain(*Z_sub))
        #X = [s[:Z_flat[i]]for i,s in enumerate(X)]
        #X = torch.tensor(pad_sequence(X).squeeze(-1), dtype=torch.float, device=self.device)

        #X = torch.tensor(pad_sequence(X).squeeze(-1),device = self.device)
        X = pad_sequence(X).squeeze(-1)
 
        X_embed = self.model_embeddings.vocabs(X)        
        out,(last_h1,last_c1) = self.sub_decoder(X_embed)
        #X_proj = torch.tanh(self.sub_de_projection(out))
        X_proj = self.sub_de_projection(out)
        #X_gate = torch.sigmoid(self.gate(torch.cat((X_embed,X_proj),-1)))
        X_gate = torch.sigmoid(self.de_gate(X_embed))
        X_way = self.dropout(X_gate * X_embed + (1-X_gate) * X_proj)      
        
        #X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
        #  torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        X_input = [torch.cat([ss[:Z[i][j]]for j,ss in enumerate(
            torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
          
        target_padded = pad_sequence(X_input).squeeze(-2)
        XO = [torch.tensor(x) for x in XO]
        #XO = pad_sequence(XO)
        XO = torch.tensor(pad_sequence(XO),device = self.device)
        #target_lengths = [sum([wl for wl in s]) for s in Z_sub]
        
        return target_padded, XO


    def parallel_decode_new(self,target, tgt_padded, lang):
    
        sbols = {'_':1,'^':2,'`':3}
        if type(target[0]) is not list: target = [target]
        Z, XO, Z_sub = get_sents_lenth4_new(target,sbols) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in target]
        max_l = max(s_len)
        #Z = [s if max_l>s_len[i] else s[:-1]+[s[-1]-1] for i,s in enumerate(Z)] 
        #Z = [[sx for sx in s if sx>0] for s in Z] 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        #max_l = max_l - 1
        XX =  [s+[max_l-s_len[i]] if max_l>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        """
        print(target[1])
        print(tgt[1])
        print("max_l :{}, tgt_padded :{}".format(max_l, tgt_padded.size()))
        print("Z_len : {}".format(Z_len))
        print("XX : {}".format(XX))
        """
        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(torch.split(tgt_padded,1,-1))]))
        #target_padded = [[w for j,w in s if target[i][j] not in sbols.keys()] for i,s in enumerate(tgt_padded)]]
        X = pad_sequence(X).squeeze(-1) 
        X_embed = self.model_embeddings.vocabs(X) 

        if lang == 'en':
            out,(last_h1,last_c1) = self.sub_en_coder(X_embed)
            X_proj = self.sub_en_projection(out[1:])
            X_gate = torch.sigmoid(self.en_gate(X_embed[1:]))
        else:
            out,(last_h1,last_c1) = self.sub_ko_coder(X_embed)
            X_proj = self.sub_ko_projection(out[1:])
            X_gate = torch.sigmoid(self.ko_gate(X_embed[1:]))

        X_way = self.dropout(X_gate * X_embed[1:] + (1-X_gate) * X_proj)      

        X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
            torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
          
        target_embedded = pad_sequence(X_input).squeeze(-2)[:-1]
        XO = [torch.tensor(x) for x in XO]
        XO = torch.tensor(pad_sequence(XO)).to(self.device) #,device = self.device) #<=[:-1]
        
        return target_embedded, XO

 
    def parallel_beam_encode(self,target):

        if type(target[0]) is not list: target = [target]
        s_len = [len(s)-1 for s in target]
        X = self.vocab.tgt.to_input_tensor(target, device=self.device)
        #tgt_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  
        #print(target)
        #X = pad_sequence(target)
        #X = torch.tensor(X,device = self.device)
 
        X_embed = torch.tensor(self.model_embeddings.target(X), device=self.device)
        #print(X_embed.shape)
 
        out,(last_h1,last_c1) = self.sub_decoder(X_embed)

        xxp = torch.tensor(s_len, device=self.device).unsqueeze(0).unsqueeze(-1)
        out = torch.gather(out, index=xxp.expand(1,-1,X_embed.size(2)), dim=0)
        #print("X_emd size {} {} {} {}".format(X_embed.shape, len(s_len), s_len[:5], xxp.shape))
        X_proj = torch.tanh(self.sub_de_projection(out))
        X_embed = torch.gather(X_embed, index=xxp.expand(1,-1,X_embed.size(2)), dim=0)
        X_gate = torch.sigmoid(self.gate(X_embed))
        X_way = X_gate * X_embed + (1-X_gate) * X_proj       
        
        return X_way

    def parallel_beam_encode2(self, X, lang, init_vecs= None):

        #X_embed = torch.tensor(self.model_embeddings.target(X), device=self.device).unsqueeze(0)
        X_embed = self.model_embeddings.vocabs(X).unsqueeze(0)
        
        #if len(X_embed.shape) <3:X_embed.unsqueeze(0)
        """
        if init_vecs is not None:
            print("X_embed / init_vec :{}, {}".format(X_embed.shape, init_vecs[0].shape))
        else:
            print("X_embed : {}".format(X_embed.shape))
        """
        if lang =='en':
            out,(h,c) = self.sub_en_coder(X_embed,init_vecs)
            #X_proj = torch.tanh(self.sub_de_projection(out))
            X_proj = self.sub_en_projection(out)
            X_gate = torch.sigmoid(self.en_gate(X_embed))

        else:
            if init_vecs is not None and len(init_vecs[0].size())<3 or len(X_embed.size()) <3:
                print("X_embed.size(), init_vecs[0].size() : {} {}".format(X_embed.size(), init_vecs[0].size()))
            out,(h,c) = self.sub_ko_coder(X_embed,init_vecs)
            #X_proj = torch.tanh(self.sub_de_projection(out))
            X_proj = self.sub_ko_projection(out)
            X_gate = torch.sigmoid(self.ko_gate(X_embed))

        X_way = (X_gate * X_embed + (1-X_gate) * X_proj).squeeze(0)
        #print("X_way : {}".format(X_way.shape))      
        
        return X_way, (h,c)

    def en_encode_origin(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        ### YOUR CODE HERE (~ 8 Lines)
        ### TODO:
        ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        ###         that there is no initial hidden state or cell for the decoder.
        ###     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        ###         - Note that the shape of the tensor returned by the encoder is (src_len b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        ###     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###         - `init_decoder_cell`:
        ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
        ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        ### See the following docs, as you may need to use some of the following functions in your implementation:
        ###     Pack the padded sequence X before passing to the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        ###     Pad the packed sequence, enc_hiddens, returned by the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Permute:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
        
         
        X = pack_padded_sequence(source_padded, source_lengths)
        out, (last_h,last_c) = self.en_encoder(X)
        enc_hiddens = pad_packed_sequence(out, batch_first=True)[0]      
        
        last_h = torch.cat((last_h[0,:],last_h[1,:]), -1)
        last_h = self.en_h_projection(last_h)
        last_c = torch.cat((last_c[0,:],last_c[1,:]), -1)
        last_c = self.en_c_projection(last_c)

        dec_init_state = (last_h, last_c)

        return enc_hiddens, dec_init_state


    def en_encode(self, source_padded: torch.Tensor, source_lengths: List[int], self_trns=0): # -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        enc_hiddens, dec_init_state = None, None
      
        X = pack_padded_sequence(source_padded, source_lengths)
        out, (last_h,last_c) = self.en_encoder(X)
        enc_hiddens = pad_packed_sequence(out, batch_first=True)[0]  #size() : (batch, len, 2*hidden)

        last_h1 = torch.cat((last_h[0,:],last_h[1,:]), -1)
        last_h1 = self.en_h1_projection(last_h1).unsqueeze(0)
        last_h2 = torch.cat((last_h[2,:],last_h[3,:]), -1)
        last_h2 = self.en_h2_projection(last_h2).unsqueeze(0) 
        last_h = torch.cat((last_h1,last_h2), 0) 

        last_c1 = torch.cat((last_c[0,:],last_c[1,:]), -1)
        last_c1 = self.en_c1_projection(last_c1).unsqueeze(0)
        last_c2 = torch.cat((last_c[2,:],last_c[3,:]), -1)
        last_c2 = self.en_c2_projection(last_c2).unsqueeze(0) 
        last_c = torch.cat((last_c1,last_c2), 0)  
     
        #last_h = torch.cat((last_h[0,:],last_h[1,:]), -1)  to_check !!!!!!!!!
        #last_h = self.en_h_projection(last_h)
        #last_c = torch.cat((last_c[0,:],last_c[1,:]), -1)
        #last_c = self.en_c_projection(last_c)

        dec_init_state = (last_h, last_c)

        #return enc_hiddens, dec_init_state
        
        if self_trns == 1:
            return self.dropout_10(enc_hiddens), dec_init_state
        else:
            return enc_hiddens, dec_init_state  
        

    def ko_encode(self, source_padded: torch.Tensor, source_lengths: List[int], self_trns=0): # -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        enc_hiddens, dec_init_state = None, None
      
        X = pack_padded_sequence(source_padded, source_lengths)
        out, (last_h,last_c) = self.ko_encoder(X)
        enc_hiddens = pad_packed_sequence(out, batch_first=True)[0]

        last_h1 = torch.cat((last_h[0,:],last_h[1,:]), -1)
        last_h1 = self.ko_h1_projection(last_h1).unsqueeze(0)
        last_h2 = torch.cat((last_h[2,:],last_h[3,:]), -1)
        last_h2 = self.ko_h2_projection(last_h2).unsqueeze(0) 
        last_h = torch.cat((last_h1,last_h2), 0) 

        last_c1 = torch.cat((last_c[0,:],last_c[1,:]), -1)
        last_c1 = self.ko_c1_projection(last_c1).unsqueeze(0)
        last_c2 = torch.cat((last_c[2,:],last_c[3,:]), -1)
        last_c2 = self.ko_c2_projection(last_c2).unsqueeze(0) 
        last_c = torch.cat((last_c1,last_c2), 0)  

        """
        last_h = torch.cat((last_h[0,:],last_h[1,:]), -1)
        last_h = self.ko_h_projection(last_h)
        last_c = torch.cat((last_c[0,:],last_c[1,:]), -1)
        last_c = self.ko_c_projection(last_c)
        """
        
        dec_init_state = (last_h, last_c)
        #return enc_hiddens, dec_init_state
        
        if self_trns == 1:
            return self.dropout_10(enc_hiddens), dec_init_state
        else:
            return enc_hiddens, dec_init_state  
        
    
    def ko_decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_embedded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size. 

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        #target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state  #(layer, batch, hidden)

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        ###         which should be shape (b, src_len, h),
        ###         where b = batch size, src_len = maximum source length, h = hidden size.
        ###         This is applying W_{attProj} to h^enc, as described in the PDF.
        ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        ###     3. Use the torch.split function to iterate over the time dimension of Y.
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e). 
        ###             - Construct Ybar_t by concatenating Y_t with o_prev.
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###   
        ### Use the following docs to implement this functionality:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     Tensor Splitting (iteration):
        ###         https://pytorch.org/docs/stable/torch.html#torch.split
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Stacking:
        ###         https://pytorch.org/docs/stable/torch.html#torch.stack
        
        enc_hiddens_proj = self.ko_att_projection(enc_hiddens)  # batch, src_len, hidden)
        #Y = self.model_embeddings.target(target_padded)
        Y = target_embedded # self.parallel_decode(target, target_padded)  # (max_len, batch, emb)
        for sp in torch.split(Y,1):
            #if len(sp.squeeze().size()) != len(o_prev.size()):
            #    print(sp.squeeze().size(),o_prev.size())
            Ybar_t = torch.cat((sp.squeeze(0), o_prev), -1)
            dec_state, o_t, e_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks, tlang='ko')
            combined_outputs.append(o_t)
            o_prev = o_t
        
        combined_outputs = torch.stack(combined_outputs) # (max_tgt_len, b, h)

        return combined_outputs

    def en_decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_embedded: torch.Tensor) -> torch.Tensor:

        dec_state = dec_init_state
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        combined_outputs = []

        enc_hiddens_proj = self.en_att_projection(enc_hiddens)
        Y = target_embedded # self.parallel_decode(target, target_padded)
        for sp in torch.split(Y,1):
            Ybar_t = torch.cat((sp.squeeze(0), o_prev), -1)
            dec_state, o_t, e_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks, tlang='en')
            combined_outputs.append(o_t)
            o_prev = o_t
        
        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs


    def ke_decode_old(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:

        target_padded = target_padded[:-1]
        dec_state = dec_init_state
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        combined_outputs = []
        
        enc_hiddens_proj = self.en_att_projection(enc_hiddens)
        Y = self.model_embeddings.vocabs(target_padded)
        for sp in torch.split(Y,1):
            #if len(sp.squeeze().size()) != len(o_prev.size()):
            #    print(sp.squeeze().size(),o_prev.size())
            Ybar_t = torch.cat((sp.squeeze(0), o_prev), -1)
            dec_state, o_t, e_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks, tlang='en')
            combined_outputs.append(o_t)
            o_prev = o_t
        
        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs    
        

    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor, tlang) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hsplt_len_flattenedidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        
        if tlang == 'en':
            _, dec_state = self.en_decoder(Ybar_t.unsqueeze(0), dec_state)
        else:
            _, dec_state = self.ko_decoder(Ybar_t.unsqueeze(0), dec_state)
        
        e_t = torch.bmm(enc_hiddens_proj, dec_state[0][1].unsqueeze(-1)).squeeze(-1) #dec_state[0][1] : h, layer2
        # (b,len,hidden) matmul (b, hidden, 1) =>  (b,len,1) => (b,len)
       
        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.splt_len_flattened
        #$$     Hints:
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh
        
        alpha_t = torch.softmax(e_t, -1).unsqueeze(-2)
        a_t = torch.bmm(alpha_t, enc_hiddens).squeeze(-2)
        #print("a_t.shape = ", a_t.shape)
        #print("dec_state[0].shape = ", dec_state[0].shape)        
        
        u_t = torch.cat((a_t, dec_state[0][1]), -1)
        if tlang == 'en':
            v_t = self.en_combined_output_projection(u_t)
        else:
            v_t = self.ko_combined_output_projection(u_t)
        O_t = self.dropout(torch.tanh(v_t))

        ### END YOUR CODEsplt_len_flattened

        combined_output = O_t
        return dec_state, combined_output, e_t, alpha_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def ek_beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70, tlang:str='en') -> List[Khypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
 
        #source_padded, source_lengths = self.parallel_encode(src_sent) 

        #src_sents_var =  source_padded
        #src_len = source_lengths[0]

        slang = 'en' if tlang == 'ko' else 'ko'

        sbolX = [torch.tensor([self.vocab.vocs[w]], dtype=torch.long, device=self.device) for w in self.sbol]      
        sbol_init = ['']+[self.parallel_beam_encode2(sb,lang=tlang) for sb in sbolX]

        src_sents_var, src_len = self.parallel_encode_new(src_sent, slang) 
        if slang == 'en':
            src_encodings, dec_init_vec = self.en_encode(src_sents_var, src_len)
        else:
            src_encodings, dec_init_vec = self.ko_encode(src_sents_var, src_len)

        if tlang =='ko':
            #src_encodings, dec_init_vec = self.en2_encode(src_sents_var, src_len)
            src_encodings_att_linear = self.ko_att_projection(src_encodings)
        else:
            #src_encodings, dec_init_vec = self.ko_encode(src_sents_var, src_len)
            src_encodings_att_linear = self.en_att_projection(src_encodings)

        """
        src_sents_var = self.vocab.vocs.to_input_tensor([src_sent], self.device)e
        #src_encodings, dec_init_vec = self.encode(src_sents_var, [src_len])
        src_encodings_att_linear = self.ko_att_projection(src_encodings)
        """
        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.vocs['</s>']

        hypotheses = [['<s>']]
        xhypotheses = [[1]]  ###??????? 나중에 지워질 것이므로 값 의미 없음 
        ahypotheses = [[]]
        uhypotheses = [[]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))  # shape (b, src_len, h*2)

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            #y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            #y_t_embed = self.model_embeddings.target(y_tm1)                                # 수정된 부분
            if t<2:
                y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)       
                y_t_embed,next_init_vecs = self.parallel_beam_encode2(y_tm1, tlang, init_vecs=sbol_init[1][1]) # '<s>' 앞의  '_' 의 초기화 
            else:
                #####xxo = torch.tensor(prev_xos, dtype=torch.float,device=self.device)[:,-1].unsqueeze(0).unsqueeze(-1) #shape(1,batch,1)
                #print( "prev_init_vecs[0][0].shape:{}".format( prev_init_vecs[0][0].shape))   
                
                prev_init_vecs = [torch.cat(vecs,0).unsqueeze(0) for vecs in prev_init_vecs]    # shape (h,c) = ((1,batch,h),(1,batch,h)
                #prev_init_vecs = [torch.cat(vecs,0).unsqueeze(0) * xxo for vecs in prev_init_vecs]
                #print( "prev_init_vecs[0].shape:{}, xxo.shape".format( prev_init_vecs[0].shape),xxo.shape)           
                y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
                y_t_embed, next_init_vecs = self.parallel_beam_encode2(y_tm1,tlang, init_vecs=prev_init_vecs)   # 수정된 부분

            #y_t_embed = self.parallel_beam_encode(y_tm1)

            #print(y_t_embed.shape, att_tm1.shape)

            #x = torch.cat([y_t_embed.squeeze(0), att_tm1], dim=-1)
            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _, alpha_t  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None, tlang=tlang)

            # log probabilities over target words
            """
            if tlang == 'ko':
                log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)   #att_t's shape = (batch,hidden)
            else:
                log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)  
            """
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)
            log_p2, xos = F.log_softmax(self.target_ox_projection(att_t), dim=-1).max(-1)
            
            log_p2 = log_p2.unsqueeze(1).expand_as(log_p_t) * self.xo_weight   # 추가된 부분
            #xos = xos.unsqueeze(0).unsqueeze(-1)  # 추가된 부분

            #print("vecs, xos : {} {}".format(next_init_vecs[0].shape, xos.shape))
            #next_init_vecs = [vecs * torch.tensor(xos,dtype=torch.float,device=self.device) for vecs in next_init_vecs]

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t+log_p2).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.vocs)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.vocs)

            next_init_vecs = [next_vecs.squeeze(0) for next_vecs in next_init_vecs]

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []
            prev_xos = []
            prev_init_vecs = [[],[]]
            new_u_scores = []
            new_a_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores ):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.vocs.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                #print("len(hypotheses):{},len(xhypotheses):{}, prev_hyp_ids :{}, xos:{}".format(
                #    len(hypotheses),len(xhypotheses),prev_hyp_ids, xos))
                new_xo = xhypotheses[prev_hyp_id] + [xos[prev_hyp_id]]
                new_u_score = uhypotheses[prev_hyp_id] + [log_p_t[prev_hyp_id,hyp_word_id].item()]
                new_a_score = ahypotheses[prev_hyp_id] + [alpha_t[prev_hyp_id]]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Khypothesis(value=new_hyp_sent[1:-1], 
                                                           xo = new_xo[1:-1],
                                                           score=cand_new_hyp_score,
                                                           u_score=new_u_score,
                                                           a_score=new_a_score)) #scalar or vector 선택할 것)
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)   # to select (h_tm1, att_tm1)'s items matching new set
                    new_hyp_scores.append(cand_new_hyp_score)
                    prev_xos.append(new_xo)
                    new_u_scores.append(new_u_score)
                    new_a_scores.append(new_a_score)
                    if new_xo[-1] < 1:
                        prev_init_vecs[0].append(next_init_vecs[0][prev_hyp_id].unsqueeze(0))  #add batch array shape(hidden) = > (1,hidden) 
                        prev_init_vecs[1].append(next_init_vecs[1][prev_hyp_id].unsqueeze(0))
                    else:
                        prev_init_vecs[0].append(sbol_init[new_xo[-1]][1][0].squeeze(0))    #  remove direction shape(1,1,hidden) =>(1,hidden)
                        prev_init_vecs[1].append(sbol_init[new_xo[-1]][1][1].squeeze(0))                            

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            #h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            h_t = h_t.transpose(1,0)[live_hyp_ids].transpose(1,0).contiguous()
            cell_t = cell_t.transpose(1,0)[live_hyp_ids].transpose(1,0).contiguous()
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            xhypotheses = prev_xos
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
            uhypotheses = new_u_scores
            ahypotheses = new_a_scores

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Khypothesis(value=hypotheses[0][1:],
                                                   xo = prev_xos[0][1:],
                                                   score=hyp_scores[0].item(),
                                                   u_score=new_u_scores[0],
                                                   a_score=new_a_scores[0]))
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        #print([w for w in completed_hypotheses[0].value][:20])
        #print([w for w in completed_hypotheses[0].xo][:20])

        best_score = -1000000.
        rL = src_len[0] * (1.06 if slang=='en' else 0.94)
        lk = rL//6
        rL = (sorted([len(hy.value) for hy in completed_hypotheses])[len(completed_hypotheses)//2]+rL)/2
        
        print('\n')
        for i,v in enumerate(completed_hypotheses): 
            snl = len(v.value)  #sentence lenth
            rpb = 1.4e-05*snl**2 - 0.0043*snl + 0.82 if slang=='en' else 1.6e-05*snl**2 - 0.0037*snl + 0.84 #repeat penalty base
            rp = 5 * min(len(list(set(v.value))) / (snl+0.0001) - rpb, 0)

            u_mean = sum([u**2 for u in v.u_score])/(snl+0.0001)
            a_mean = 1.0*(sum(torch.log(torch.min(sum(v.a_score),torch.ones((1,src_len[0])).to(self.device)).squeeze(0))/src_len[0]))
            
            #to_check = 1 * v.score/(snl+0.0001) - 4* a_mean - 1.0 * u_mean + rp
            bx= 1.1
            to_check = 1 * v.score/(max(bx-1,0.001)/bx+snl/bx)**1.0 + 0.9* a_mean  #from GNMT
            
            #print(f"to_check = {to_check}, score = {1 * v.score/(snl+0.0001)}, u_mean = {u_mean}, a_mean = {a_mean}, rp = {rp}")

            #to_check =  v.score/(snl+0.0001) + (-4*(snl/rL-1)**2 if snl/rL<1 else -0.05*(snl/rL-1)**2) + rp
            #to_check =  v.score/(snl+0.0001) + (-2*(snl/rL-1)**2+(snl/rL-1)*0.4 if snl/rL<1 else -0.2*(snl/rL-1)**2+(snl/rL-1)*0.4) + rp
            #to_check = v.score/(snl+0.0001) + 0.7*len(list(set(v.value))) /(snl+0.0001) + 1.0 * ((snl*2/src_len[0])**0.4-1)
            
            #to_check =  v.score/(len(v.xo)+0.0001) + 0.85*min(1-rL/(len(v.value)+0.0001),0)
            #bp = [min(1-rL/(len(v.value)+il+0.0001),0) for il in list(range(-ik,ik+1,1)) if len(v.value)+il>-1]
            #to_check =  v.score/(len(v.xo)+0.0001) + 0.3*sum(bp)/len(bp) #min(1-rL/(len(v.value)+0.0001),0)
            #if v.score/len(v.xo) > best_score:
            #if completed_hypotheses[i].score/len(completed_hypotheses[i].xo) > best_score:
            #print("v.score {}, len of v.xo {} len of set {} src len {}".format(v.score, len(v.xo), len(list(set(v.value))), src_len[0]))
            ###to_check = v.score/(len(v.xo)+0.0001) + 0.7*len(list(set(v.value))) / len(v.value) + 1.0 * ((len(v.value)*2/src_len[0])**0.4-1)
            #to_check =  v.score/(len(v.xo)+0.0001) + (1.0*len(list(set(v.value))) / (len(v.value)+0.0001) + 0.5 * ((len(v.value)*2/(src_len[0]+0.0001))**0.4-1)) * min(src_len[0]/15,1)
            if to_check > best_score: 
                #best_score = completed_hypotheses[i].score/len(completed_hypotheses[i].xo)
                #best_score = v.score/(len(v.xo)+0.0001)
                best_score = to_check
                k_best = i
        completed_hypotheses[0] = completed_hypotheses[k_best]

        temp_h = []
        #pre_v = ''
        for i,v in enumerate(completed_hypotheses[0].value):
           if i>0 and completed_hypotheses[0].xo[i] == 0:
              temp_h += [v]
           #elif i>0 and pre_v not in self.token and v not in self.token and completed_hypotheses[0].xo[i] == 0:
           #   temp_h += ['_'] + [v]
           else:
              temp_h += [self.sbol[completed_hypotheses[0].xo[i]-1]]+[v]
           #pre_v = v
        completed_hypotheses[0] = Khypothesis(value=temp_h,
                                             xo =  completed_hypotheses[0].xo,
                                             score= completed_hypotheses[0].score,
                                             u_score = completed_hypotheses[0].u_score,
                                             a_score = completed_hypotheses[0].a_score)

        return completed_hypotheses

    
    def ke_beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Ehypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
 
        src_sents_var, src_len = self.parallel_encode(src_sent) 

        #src_sents_var =  source_padded
        #src_len = source_lengths[0]
 
        #src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        #src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings, dec_init_vec = self.ko_encode(src_sents_var, src_len)
        src_encodings_att_linear = self.en_att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.vocs['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.vocabs(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _, alpha_t  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None, tlang='en')

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.vocs)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.vocs)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.vocs.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Ehypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Ehypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses    

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        #return self.model_embeddings.source.weight.device
        return self.model_embeddings.vocabs.weight.device  ##################################


    @staticmethod
    def load(model_path: str, char_size: int, wid2cid: dict):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], char_size=char_size, wid2cid=wid2cid, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


    def greedy_search(self, src_sent: List[str], max_decoding_time_step: int=70, slang:str='en', tlang:str='ko', mapping:int=0) -> List[Khypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
 
        #source_padded, source_lengths = self.parallel_encode(src_sent) 

        #src_sents_var =  source_padded
        #src_len = source_lengths[0]
        #slang = 'en' if tlang == 'ko' else 'ko'

        slen = [(i, len([k for k in s if k not in self.sbol])) for i,s in enumerate(src_sent)]
        slen = sorted(slen, key = lambda x:x[1], reverse=True)
        src_sent = [src_sent[s[0]] for s in slen]
        sent_order = [ss[0] for ss in sorted([(i,s[0]) for i,s in enumerate(slen)], key = lambda x:x[1])]
        #  1:3     2 7    1 2     3
        #  2:7  => 4 5 => 2 4 =>  1
        #  3:2     1 3    3 1     4   =>sent_order = [3,1,4,2]
        #  4:5     3 2    4 3     2

        sbolX = [torch.tensor([self.vocab.vocs[w]], dtype=torch.long, device=self.device) for w in self.sbol]      
        sbol_init = ['']+[self.parallel_beam_encode2(sb, tlang) for sb in sbolX]
        #print("mapping : {}".format(mapping))
        src_sents_var, src_len = self.parallel_encode_new(src_sent, slang, mapping=mapping) 
        if slang == 'en':
            src_encodings, dec_init_vec = self.en_encode(src_sents_var, src_len)  
        else:   
            src_encodings, dec_init_vec = self.ko_encode(src_sents_var, src_len)       

        if tlang =='ko':
            src_encodings_att_linear = self.ko_att_projection(src_encodings)
        else:
            src_encodings_att_linear = self.en_att_projection(src_encodings)

        enc_masks = self.generate_sent_masks(src_encodings, src_len)

        """
        src_sents_var = self.vocab.vocs.to_input_tensor([src_sent], self.device)e
        #src_encodings, dec_init_vec = self.encode(src_sents_var, [src_len])
        src_encodings_att_linear = self.ko_att_projection(src_encodings)
        """
        h_tm1 = dec_init_vec
        b_size = src_encodings.size(0)
        att_tm1 = torch.zeros(b_size, self.hidden_size, device=self.device)

        eos_id = self.vocab.vocs['</s>']

        #hypotheses = [['<s>'] * b_size]
        sents = [['<s>']] * b_size
        #print("sents : {}".format(len(sents)))
        xos = [[1]] * b_size  ### '1' 은 사용되지 않고 버려짐 xos[1:-1]
        #sents_id = [[[1]] * b_size]
        live_sent_ids = torch.arange(b_size)
        scores = torch.zeros(b_size, dtype=torch.float, device=self.device)
        #completed_hypotheses = []
        batch_done = [{} for i in range(b_size)]

        t = 0
        while len(live_sent_ids) > 0: # and t < max_decoding_time_step:
            t += 1
            #hyp_num = len(hypotheses)
            """
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))  # shape (b, src_len, h*2)

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))
            """
            #y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            #y_t_embed = self.model_embeddings.target(y_tm1)                                # 수정된 부분
            if t<2:
                y_tm1 = torch.tensor([self.vocab.vocs[s[-1]] for s in sents], dtype=torch.long, device=self.device) #.squeeze(-1)      
                #print("y_tm1.size()  : {}, sents[0][-1]: {}".format(y_tm1.size(), sents[0][-1]))
                pre_init_vecs=[sb.expand(1,b_size,self.hidden_size).contiguous() for sb in sbol_init[1][1]]

                y_t_embed,next_init_vecs = self.parallel_beam_encode2(y_tm1, tlang, init_vecs=pre_init_vecs)
            else:
                #####xxo = torch.tensor(prev_xos, dtype=torch.float,device=self.device)[:,-1].unsqueeze(0).unsqueeze(-1) #shape(1,batch,1)
                #print( "prev_init_vecs[0][0].shape:{}".format( prev_init_vecs[0][0].shape))   
                if len(prev_init_vecs[0]) >1:
                    prev_init_vecs = [torch.cat(vecs,0).unsqueeze(0).contiguous() for vecs in prev_init_vecs]    # shape (h,c) = ((1,batch,h),(1,batch,h)
                else:
                    prev_init_vecs = [vecs[0].unsqueeze(0).contiguous() for vecs in prev_init_vecs] 
                #prev_init_vecs = [torch.cat(vecs,0).unsqueeze(0) * xxo for vecs in prev_init_vecs]
                #print( "prev_init_vecs[0].shape:{}, xxo.shape".format( prev_init_vecs[0].shape),xxo.shape)           
                y_tm1 = torch.tensor([self.vocab.vocs[s[-1]] for s in sents], dtype=torch.long, device=self.device)
                y_t_embed, next_init_vecs = self.parallel_beam_encode2(y_tm1,tlang, init_vecs=prev_init_vecs)   # 수정된 부분

            #y_t_embed = self.parallel_beam_encode(y_tm1)

            #print(y_t_embed.shape, att_tm1.shape)

            #x = torch.cat([y_t_embed.squeeze(0), att_tm1], dim=-1)
            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _, alpha_t  = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear, enc_masks=enc_masks, tlang=tlang)

            # log probabilities over target words
            log_p_t, wid = F.log_softmax(self.target_vocab_projection(att_t), dim=-1).max(-1)
            """
            if tlang == 'ko':
                log_p_t, wid = F.log_softmax(self.target_vocab_projection(att_t), dim=-1).max(-1)   #att_t's shape = (batch,hidden)
            else:
                log_p_t, wid = F.log_softmax(self.target_vocab_projection(att_t), dim=-1).max(-1)  
            """
            log_p2, xo = F.log_softmax(self.target_ox_projection(att_t), dim=-1).max(-1)  
            log_p2 = log_p2 * self.xo_weight          
            #log_p2 = log_p2.unsqueeze(1).expand_as(log_p_t) * self.xo_weight   # 추가된 부분
            #xos = xos.unsqueeze(0).unsqueeze(-1)  # 추가된 부분

            #print("vecs, xos : {} {}".format(next_init_vecs[0].shape, xos.shape))
            #next_init_vecs = [vecs * torch.tensor(xos,dtype=torch.float,device=self.device) for vecs in next_init_vecs]

            #live_hyp_num = b_size - len(batch_done)
            scores = (scores + log_p_t+log_p2)
            #contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t+log_p2).view(-1)
            # top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            #prev_hyp_ids = list(range(len(wid))) #top_cand_hyp_pos / len(self.vocab.vocs)
            #hyp_word_ids = wid #top_cand_hyp_pos % len(self.vocab.vocs)
            next_init_vecs = [next_vecs.squeeze(0) for next_vecs in next_init_vecs]
            sents  = [sents[i]+[self.vocab.vocs.id2word[wid[i].item()]] for i in range(len(live_sent_ids))]
            #print("b_size :{},len(xos) : {}, wid.size() : {}, xo.size() : {}".format(b_size,len(xos),wid.size(), xo.size()))
            xos = [xos[i] + [xo[i].item()] for i in range(len(live_sent_ids)) ]

            prev_init_vecs = [[],[]]

            if eos_id not in wid and t < max_decoding_time_step:
                for i in range(len(live_sent_ids)):
                    if xo[i].item() < 1:
                        prev_init_vecs[0].append(next_init_vecs[0][i].unsqueeze(0))  #add batch array shape(hidden) = > (1,hidden) 
                        prev_init_vecs[1].append(next_init_vecs[1][i].unsqueeze(0))
                    else:
                        prev_init_vecs[0].append(sbol_init[xo[i].item()][1][0].squeeze(0))    #  remove direction shape(1,1,hidden) =>(1,hidden)
                        prev_init_vecs[1].append(sbol_init[xo[i].item()][1][1].squeeze(0)) 
                h_tm1 = (h_t, cell_t)
                att_tm1 = att_t

            else:
                #prev_init_vecs = [[],[]]
                new_live_sent_ids = []
                new_temp_ids = []
                new_sents = []
                new_xos = []

                for i in range(len(live_sent_ids)):
                    b_id = live_sent_ids[i]
                    if wid[i].item() == eos_id or t == max_decoding_time_step:                       
                        batch_done[b_id] = {'sent':sents[i][1:-1], 'xo':xos[i][1:-1], 'score':scores[i].item()}
                    else:
                        new_live_sent_ids.append(b_id)
                        new_temp_ids.append(i)
                        new_sents.append(sents[i])
                        new_xos.append(xos[i])

                        if xo[i].item() < 1:
                            prev_init_vecs[0].append(next_init_vecs[0][i].unsqueeze(0))  #add batch array shape(hidden) = > (1,hidden) 
                            prev_init_vecs[1].append(next_init_vecs[1][i].unsqueeze(0))
                        else:
                            prev_init_vecs[0].append(sbol_init[xo[i].item()][1][0].squeeze(0))    #  remove direction shape(1,1,hidden) =>(1,hidden)
                            prev_init_vecs[1].append(sbol_init[xo[i].item()][1][1].squeeze(0)) 

                live_ids = torch.tensor(new_temp_ids, dtype=torch.long, device=self.device)
                #h_tm1 = (h_t[live_ids], cell_t[live_ids])
                h_tm1 = (torch.cat((h_t[0][live_ids].unsqueeze(0),h_t[1][live_ids].unsqueeze(0)),0),
                         torch.cat((cell_t[0][live_ids].unsqueeze(0),cell_t[1][live_ids].unsqueeze(0)),0))
                att_tm1 = att_t[live_ids]
                src_encodings = src_encodings[live_ids]
                src_encodings_att_linear = src_encodings_att_linear[live_ids]
                enc_masks = enc_masks[live_ids]
                live_sent_ids = new_live_sent_ids
                sents = new_sents
                xos = new_xos
                scores = scores[live_ids]

            if len(live_sent_ids) == 0:
                break

        sents = []
        scores = []
        #print(len(batch_done),batch_done[1])
        for b in batch_done:

            sent = b['sent']
            xo = b['xo']

            #if len(sent) < 1: continue

            X = []
            #pre_v = ''
            if len(xo) < 1:
                xo = [1]
                sent = ['a']
                b['score'] = -100.0
                print("xo : {}, sent : {}, len(batch_done) : {}".format(xo, sent,len(batch_done)))
            if xo[0] ==0:
                print("Starting word sbol is empty space !!!!")
                xo[0] = 1
            for i,v in enumerate(sent):
                #if v in self.sbol:
                #    continue
                if xo[i] == 0:
                    X += [v]
                elif v not in ['', ' ', '_', '^', '`']: #else:
                    X += [self.sbol[xo[i]-1], v]
            sents.append(' '.join(X))
            scores.append(b['score']/len(sent))

        sents = [sents[i] for i in sent_order]
        scores = [scores[i] for i in sent_order]
        #print(sents[0])

        return sents, scores
