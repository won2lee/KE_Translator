import torch
from trns.vocab import Vocab, get_wid2cid
from trns.nmt_model import NMT
from trns.preproc_En import Pre_en
from trns.preproc_kor import preproc_ko2en
from trns.trns_koren import Trns

def trns_model():
    pre_en = Pre_en()
    pre_ko = preproc_ko2en()
    vocab_trns = Vocab.load('trns/vocab.json')
    wid2cid = get_wid2cid()
    model = NMT(vocab=vocab_trns, embed_size=300, hidden_size=300, char_size=85, wid2cid=wid2cid, dropout_rate=0.0)
    model.load_state_dict(torch.load('trns/model_bi_1105', map_location=lambda storage, loc: storage))
    model.eval()
    trns = Trns(model)
    
    return pre_en, pre_ko, trns
