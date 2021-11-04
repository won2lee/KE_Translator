

import re
import collections
from collections import Counter


def json_save(data, f_name):
    import json
    json = json.dumps(data)
    with open(f_name +".json","w") as f:
        f.write(json) 

def json_read(f_name):
    import json
    with open(f_name) as f:
        return json.load(f)


def get_filelist():
    #path = '../wikiextractor/text/AA/wiki_'
    path = '../wikiextractor/text/A'
    fileN = '0123456789'
    fileABC = 'ABCDEFG'
    #file_list = "subtxt.txt,subtxt2.txt".split(',')
    file_list = []
    for a in fileABC:
        for n in fileN:
            if (a=='G') and (n=='5'):
                file_list += [path+a+'/wiki_' + n + s for s in fileN[:9]]
                break
            else: 
                file_list += [path+a+'/wiki_' + n + s for s in fileN]
    return file_list     


def get_text(file):
    with open(file, 'r') as f:
        data = f.read()
    p = re.compile(r'\<.*\>')
    #q = re.compile(r'[^0-9a-zA-Z가-힣\-]')
    #######################################################################################
    q = re.compile(r'[^0-9가-힣\-]') #'_' 가 들어갈 경우 별도조치 필요 : start_ch 와 중복!!!!!!!!!!
    #######################################################################################
    d_out = p.sub(' ', data)
    d_out = q.sub(' ', d_out)
    return re.sub('\s{1,}',' ', d_out)  # sentence list

def get_sentences(file):
    with open(file, 'r') as f:
        data = f.read()

    p = re.compile(r'\<.*\>')
    d_out = p.sub(' ', data)

    q = re.compile(r'[^0-9a-zA-Z가-힣\-\.]')
    d_out = q.sub(' ', d_out)
    d_out = re.sub('\s{1,}',' ', d_out)

    
    sentences = d_out.split('. ')
    return [re.sub('\.',' ',s) for s in sentences]

def sentences_from_filelist(file_list, num_files):
    sentences =[]
    for f in file_list[:num_files]:    
        sentences +=  get_sentences(f)
    return sentences

def count_from_file(file_list):
    c = Counter()
    for f in file_list:
        d_out =  get_text(f)
        c = c + Counter(d_out.split(' '))
    return c



"""
kor_alpha = CHOSUNG_LIST + ['#']*(ch_len - len(CHOSUNG_LIST)) + JUNGSUNG_LIST + ['#']*(ch_len - len(JUNGSUNG_LIST))+JONGSUNG_LIST
kor_alpha = kor_alpha+['_','$']
etc = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 
       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
       'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 
       'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
       '{', '|', '}', '~']
#alpha = [C for C in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]+[c for c in 'abcdefghijklmnopqrstuvwxyz']
kor_alpha += etc
"""

# 스페셜 글자를 정상적인 글자로 전환
def special_to_normal(s,key_vars):
    BASE_CODE = key_vars['BASE_CODE']

    return ind_char([ord(c)-BASE_CODE for c in s], key_vars)  


def normal_to_special(word, key_vars):
    
    BASE_CODE = key_vars['BASE_CODE']
    
    special = []
    for i in convert(word,key_vars)[1]:
        if i != ' ':
            special.append(chr(i+BASE_CODE))
        else:
            special.append(' ')       
    
    return ''.join(special)
    

#######################################################
#  자소 인덱스를 글자로 변환하여 글자를 식별할 수 있도록 
#######################################################

def ind_char(idx, key_vars):
    
    ch_len = key_vars['ch_len']
    CHOSUNG = key_vars['CHOSUNG']
    JUNGSUNG = key_vars['JUNGSUNG']
    BASE_CODE = key_vars['BASE_CODE']
    kor_alpha = key_vars['kor_alpha']
   
    word_comb = []
    nc = [CHOSUNG, JUNGSUNG, 1]
    w_idx = BASE_CODE
    n_ind = 0
    
    for i,id in enumerate(idx):
        if id > 83:
            #print(id)
            word_comb.append(kor_alpha[id])
            w_idx = BASE_CODE
            n_ind = 0
            continue
        if id<0:
            word_comb.append(' ')
            w_idx = BASE_CODE
            n_ind = 0
            continue
            
        w_idx += nc[id//ch_len] * (id%ch_len)    #초,중,종성 구분 + 각 분류에서 몇번째 인지
        n_ind += 1
        if i == len(idx)-1:                      #맨 마지막이면..
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            #n_ind = 0
        elif (idx[i+1]//ch_len==0) or (idx[i+1] < 0): # or (idx[i+1]>ch_len*3):             한 글자가 끝나면..
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            w_idx = BASE_CODE
            n_ind = 0

    return ''.join(word_comb) 


def convert(test_keyword, key_vars):
    
    ch_len = key_vars['ch_len']
    CHOSUNG = key_vars['CHOSUNG']
    JUNGSUNG = key_vars['JUNGSUNG']
    BASE_CODE = key_vars['BASE_CODE']
    kor_alpha = key_vars['kor_alpha'] 
    CHOSUNG_LIST  = key_vars['CHOSUNG_LIST'] 
    JUNGSUNG_LIST = key_vars['JUNGSUNG_LIST']
    JONGSUNG_LIST = key_vars['JONGSUNG_LIST']    
        
    split_keyword_list = list(test_keyword)
    #print(split_keyword_list)

    result = list()
    indexed = list()
    num_etc = {}
    for i,s in enumerate('0123456789-_'):
        num_etc[s] = i+84
    
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - BASE_CODE
            char1 = int(char_code / CHOSUNG)
            indexed.append(char1)
            result.append(CHOSUNG_LIST[char1])
            #print('초성 : {}'.format(CHOSUNG_LIST[char1]))
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            indexed.append(char2 + ch_len)
            result.append(JUNGSUNG_LIST[char2])
            #print('중성 : {}'.format(JUNGSUNG_LIST[char2]))
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            
            if char3==0:
                result.append('#')
            else:
                result.append(JONGSUNG_LIST[char3])
                indexed.append(char3 +ch_len*2)
            #print('종성 : {}'.format(JONGSUNG_LIST[char3]))
        else:
            result.append(keyword)
            if keyword in '0123456789-_':
                 indexed.append(num_etc[keyword])
            elif keyword == ' ':
                indexed.append(keyword)
                
            #if ord(keyword) <127:
            #    indexed.append(ord(keyword)+54)  # 한글이 아닌 경우 잠정적으로 +54  숫자, 영문 대소문자 포함
               
                
    # result
    return "".join(result), indexed    #indexed는 자음(초성과 종성 구분), 모음 구분하여 인덱스 각 => [ㄱ ㅏ ㄱ], [0,28,57]
  



#def vocab_initialize(counters,BASE_CODE,ch_start,ch_end):

def vocab_initialize(counters,key_vars):
    
    BASE_CODE = key_vars['BASE_CODE']
    ch_start = key_vars['ch_start']
    
    vocab = {}
    #for k,v in counters.items():
    #####################
    p = re.compile(r"(?P<num1>[걔-걝]+)\s(?P<num2>[걔-걝]+)")
    #####################
    
    for k,v in counters:
        _, idx = convert(k,key_vars)
        #vocab[' '.join([ch_start]+[chr(i+BASE_CODE) for i in idx])] = v
        
        ###################
        # 숫자 합치기
        #p = re.compile(r"(?P<num1>[걔-걝]+)\s(?P<num2>[걔-걝]+)")
        kw = ' '.join([ch_start]+[chr(i+BASE_CODE) for i in idx])
            
        while p.search(kw):
            kw = p.sub('\g<num1>\g<num2>', kw) 
      
        vocab[kw] = v
                     
        ################### 
                     
        #vocab[' '.join([ch_start]+[chr(i+BASE_CODE) for i in idx] + [ch_end])] = v 
    return vocab


def number_attach(counters):

    vocab = {}
    p = re.compile(r"(?P<num1>[걔-걝]+)\s(?P<num2>[걔-걝]+.*)")  
    q = re.compile(r"(?P<num3>[걟][걔-걝]*)\s(?P<num4>[걔-걝]+.*)")
    num_att = []  # 체크용
    for kw,v in counters.items():
        i = 0 # 체크용
        while p.search(kw):
            kw = p.sub('\g<num1>\g<num2>', kw) 
            i+=1  # 체크용
        kw = q.sub('\g<num3>\g<num4>', kw)
        vocab[kw] = v
        num_att.append(i) # 체크용
    print(len(num_att), sum([1 for i in num_att if i>0]), max(num_att)) 
    return vocab




def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
""" 
vocab = {'l o w </w>' : 5,
         'l o w e r </w>' : 2,
         'n e w e s t </w>':6,
         'w i d e s t </w>':3
         }
"""


def dpe_iteration(vocab, iter_num):
    num_merges = iter_num
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        #print(best)
    return vocab    
 
    
def voc_combined(vocab):
    vocab_sub = {}
    #for k, v in vocab:
    for k, v in vocab.items():
        for s in k.split(' '): 
            if s in vocab_sub.keys():
                vocab_sub[s] += v
            else:
                vocab_sub[s] = v
    return vocab_sub, len(vocab_sub)





def vocab_select(vocabs,path,iter_size, step_size,step_from_start,save_cyc):
    pre_voc_volume = 0
    n = 0
    for i in range(iter_size//step_size):
        
        vocabs = dpe_iteration(vocabs,step_size)
        _, voc_volume = voc_combined(vocabs) 
        print("iter_number :{}, voc_volume : {}".format((i+1)*step_size+step_from_start, voc_volume))
        iter_n = (i+1) * step_size 
        if iter_n % save_cyc == 0:
            json_save(vocabs, path+ str(step_from_start+iter_n))
           
        if voc_volume < pre_voc_volume:n+=1
        pre_voc_volume = voc_volume
        if n>3:break
       
    step_from_start += iter_size
    #json_save(vocabs, path+'vocabs_'+ str(step_from_start))
    #print("iter_number :{}, voc_volume : {}".format(step_from_start, voc_volume))
    
    return vocabs, voc_volume, step_from_start

#iter_size, step_size,step_from_start = 6,2,0   
#vocs, v_vol, from_start = vocab_select(vocab, iter_size, step_size,step_from_start)  





