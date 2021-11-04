import copy
#from trns.NMT.xutils_for_sents_preproc import log_prob_by_len3,to_bpe_sents10, save_sents
from trns.NMT.xutils_for_key_vars import make_key_vars
#from trns.NMT.utils_etc import filter_chin_num, json_read, json_save


from trns.NMT.xutils import json_read  #, json_save, voc_combined, special_to_normal
#from counter_vocab_tuning import dict_merge
from trns.NMT.xutils_for_sents_v2 import log_prob_by_len3, to_bpe_sents10
from trns.NMT.xutils_to_save import save_sents


def get_data():
    path = 'trns/NMT/Data/'
    vocabs = json_read(path+'vocabs.json')
    extracted_vocs = json_read(path+'modified_extracted.json')
    en_vocs = json_read(path+'en_vocabs_to_apply_0615.json')

    key_vars = make_key_vars()

    logprob = log_prob_by_len3(vocabs)
    sum_voc_vals = sum(vocabs.values())
    
    vocs_dict = [{},{}]
    vocs_dict[0] = copy.deepcopy(vocabs)
    vocs_dict[1] = copy.deepcopy(vocabs)
    for i in range(2):
        for k,l in vocs_dict[i].items():
            vocs_dict[i][k] = k          
           
    return vocabs,vocs_dict, logprob, key_vars, sum_voc_vals,extracted_vocs, en_vocs

class Preproc(object):
    
    def __init__(self, vocabs,vocs_dict, logprob, key_vars, sum_voc_vals,extracted_vocs, en_vocs):
        self.vocabs = vocabs
        self.vocs_dict = vocs_dict
        self.logprob = logprob
        self.key_vars = key_vars
        self.sum_voc_vals = sum_voc_vals
        self.extracted_vocs = extracted_vocs
        self.en_vocs = en_vocs
    
    def forward(self, X):          
        X = [''.join([c for c in s if ord(c) < 55204]) for s in X]
        X = to_bpe_sents10(X, self.vocabs, self.vocs_dict, self.logprob, self.key_vars, 
                           self.sum_voc_vals,self.extracted_vocs,(1.3,3.2,20,1),(6,5)) #default (1,3,12)
        X = save_sents(X, self.en_vocs)
        return X  
    
def preproc_ko2en():

    vocabs,vocs_dict, logprob, key_vars, sum_voc_vals,extracted_vocs, en_vocs = get_data()
    pre_fn = Preproc(vocabs,vocs_dict, logprob, key_vars, sum_voc_vals,extracted_vocs, en_vocs)

    return pre_fn
